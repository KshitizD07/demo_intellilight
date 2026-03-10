"""
IntelliLight SUMO Environment Module
=====================================

Handles SUMO simulation lifecycle management including:
- Starting and stopping SUMO processes
- TraCI API communication
- Process cleanup and error recovery
- Data collection from simulation

This module provides a clean interface to SUMO, isolating all
simulation-specific logic from the RL environment.
"""

import os
import sys
import time
import subprocess
import logging
from typing import Optional, List, Dict, Tuple
import psutil

try:
    import traci
except ImportError:
    raise ImportError(
        "TraCI not found. Please install SUMO and ensure it's in your PATH. "
        "Visit: https://www.eclipse.org/sumo/"
    )

from configs.parameters import (
    SUMOConfig,
    NetworkTopology,
    EpisodeConfig,
    Paths
)

# Setup logging
logger = logging.getLogger(__name__)


class SUMOSimulation:
    """
    Manages SUMO simulation lifecycle and TraCI communication.
    
    This class encapsulates all SUMO-specific operations, providing
    a clean interface for the RL environment to interact with traffic simulation.
    """
    
    def __init__(
        self,
        sumo_cfg: Optional[str] = None,
        use_gui: bool = False,
        step_length: float = 1.0,
        timeout: int = 30
    ):
        """
        Initialize SUMO simulation manager.
        
        Args:
            sumo_cfg: Path to SUMO configuration file (defaults to config)
            use_gui: Whether to use SUMO GUI for visualization
            step_length: Simulation time step in seconds
            timeout: Timeout for SUMO operations in seconds
        """
        self._is_running = False
        self._current_step = 0
        self._route_file = None
        self._sumo_process = None
        self.cummulative_arrived = 0
        self._last_reported_arrivals = 0

        self.sumo_cfg = sumo_cfg or os.path.join(
            Paths.SUMO_CONFIG_DIR,
            SUMOConfig.CONFIG_FILE
        )
        self.use_gui = use_gui
        self.step_length = step_length
        self.timeout = timeout
        
        # Determine SUMO binary
        self.sumo_binary = "sumo-gui" if use_gui else SUMOConfig.BINARY
        
        # Verify SUMO binary exists
        if not self._check_sumo_available():
            raise RuntimeError(
                f"SUMO binary '{self.sumo_binary}' not found in PATH. "
                "Please install SUMO and add it to your system PATH."
            )
        
        # Verify config file exists
        if not os.path.exists(self.sumo_cfg):
            raise FileNotFoundError(
                f"SUMO config file not found: {self.sumo_cfg}\n"
                f"Please ensure SUMO network files are in {Paths.SUMO_CONFIG_DIR}"
            )
        
        # # Simulation state
        # self._is_running = False
        # self._current_step = 0
        # self._route_file = None
        # self._sumo_process = None
        
        logger.info(
            f"SUMO simulation manager initialized: "
            f"binary={self.sumo_binary}, gui={use_gui}"
        )
    
    def _check_sumo_available(self) -> bool:
        """
        Check if SUMO is available in system PATH.
        
        Returns:
            bool: True if SUMO binary is available
        """
        try:
            # Try to run SUMO with --version flag
            result = subprocess.run(
                [self.sumo_binary, "--version"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def start(self, route_file: str):
        """
        Start SUMO simulation with specified route file.
        
        Args:
            route_file: Path to SUMO route file (.rou.xml)
        
        Raises:
            RuntimeError: If SUMO fails to start
            FileNotFoundError: If route file doesn't exist
        """
        if self._is_running:
            # logger.warning("SUMO already running, closing previous instance")
            self.close()
        
        if not os.path.exists(route_file):
            raise FileNotFoundError(f"Route file not found: {route_file}")
        
        self._route_file = route_file
        
        # Build SUMO command
        sumo_cmd = self._build_command(route_file)
        logger.debug(f"Starting SUMO: {' '.join(sumo_cmd)}")
        
        try:
            # FIX: Aggressively clean up ANY zombie connections before attempting to start
            try:
                traci.close()
            except Exception:
                pass

            start_time = time.time()
            
            # Start SUMO with traci
            traci.start(sumo_cmd)
            
            # Verify connection
            if not traci.isLoaded():
                raise RuntimeError("SUMO started but TraCI not connected")
            
            self._is_running = True
            self._current_step = 0
            self.cummulative_arrived = 0
            
            startup_time = time.time() - start_time
            logger.info(f"SUMO started successfully in {startup_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to start SUMO: {e}")
            
            # FIX: Ensure the TraCI connection is cleared even if SUMO crashes heavily
            try:
                traci.close()
            except Exception:
                pass
                
            self._is_running = False
            raise RuntimeError(f"SUMO startup failed: {e}")
    
    def _build_command(self, route_file: str) -> List[str]:
        """
        Build SUMO command line with all necessary parameters.
        
        Args:
            route_file: Path to route file
        
        Returns:
            list: Complete SUMO command
        """
        cmd = [
            self.sumo_binary,
            "-c", self.sumo_cfg,
            "--step-length", str(self.step_length),
            "--route-files", route_file,
            "--quit-on-end"
        ]
        
        # Add optimization flags for headless mode
        if not self.use_gui:
            cmd.extend(SUMOConfig.FLAGS)
        
        return cmd
    
    def step(self) -> None:
        """
        Advance simulation by one time step.
        
        Raises:
            RuntimeError: If SUMO is not running or step fails
        """
        if not self._is_running:
            raise RuntimeError("SUMO is not running. Call start() first.")
        
        try:
            traci.simulationStep()
            self._current_step += 1
            arrived_this_step=len(traci.simulation.getArrivedIDList())
            self.cummulative_arrived+=arrived_this_step

            if self.cummulative_arrived < self._last_reported_arrivals:
                # print(f"[ERROR] Cumulative counter went backwards! " f"{self._last_reported_arrivals} -> {self.cummulative_arrived}")
                self.cummulative_arrived = self._last_reported_arrivals

            # if arrived_this_step > 0:
            #     print(f"[SUMO] Step {self._current_step}: +{arrived_this_step} arrived, total={self.cummulative_arrived}")
            
        except traci.TraCIException as e:
            logger.error(f"SUMO step failed: {e}")
            self._is_running = False
            raise RuntimeError(f"Simulation step failed: {e}")
    
    def get_current_time(self) -> float:
        """
        Get current simulation time.
        
        Returns:
            float: Current simulation time in seconds
        """
        if not self._is_running:
            return 0.0
        
        try:
            return traci.simulation.getTime()
        except traci.TraCIException:
            return float(self._current_step * self.step_length)
    
    def get_departed_vehicles(self) -> int:
        """
        Get number of vehicles that departed in the last step.
        
        Returns:
            int: Number of departed vehicles
        """
        if not self._is_running:
            return 0
        
        try:
            return len(traci.simulation.getDepartedIDList())
        except traci.TraCIException as e:
            logger.warning(f"Failed to get departed vehicles: {e}")
            return 0
    
    def get_arrived_vehicles(self) -> int:
        """
        Get cumulative number of vehicles that arrived since episode start.
        
        This tracks ALL vehicles that completed their journey during the episode,
        not just those that arrived in the last simulation step.
        
        Returns:
            int: Total cumulative arrived vehicles
        """
        if not self._is_running:
            return 0
        
        # Debug: Track for safety
        self._last_reported_arrivals = self.cummulative_arrived
        
        # Debug print
        # print(f"[SUMO.get_arrived] Returning cumulative: {self.cummulative_arrived}")
        
        return self.cummulative_arrived
    
    def get_vehicle_count(self) -> int:
        """
        Get total number of vehicles currently in simulation.
        
        Returns:
            int: Current vehicle count
        """
        if not self._is_running:
            return 0
        
        try:
            return traci.vehicle.getIDCount()
        except traci.TraCIException as e:
            logger.warning(f"Failed to get vehicle count: {e}")
            return 0
    
    def get_lane_vehicle_count(self, lane_id: str) -> int:
        """
        Get number of vehicles on a specific lane.
        
        Args:
            lane_id: SUMO lane ID
        
        Returns:
            int: Number of vehicles on lane
        """
        if not self._is_running:
            return 0
        
        try:
            return traci.lane.getLastStepVehicleNumber(lane_id)
        except traci.TraCIException:
            return 0
    
    def get_lane_waiting_time(self, lane_id: str) -> float:
        """
        Get total waiting time on a specific lane.
        
        Args:
            lane_id: SUMO lane ID
        
        Returns:
            float: Total waiting time in seconds
        """
        if not self._is_running:
            return 0.0
        
        try:
            return traci.lane.getWaitingTime(lane_id)
        except traci.TraCIException:
            return 0.0
    
    def get_lane_mean_speed(self, lane_id: str) -> float:
        """
        Get mean speed of vehicles on a lane.
        
        Args:
            lane_id: SUMO lane ID
        
        Returns:
            float: Mean speed in m/s
        """
        if not self._is_running:
            return 0.0
        
        try:
            return traci.lane.getLastStepMeanSpeed(lane_id)
        except traci.TraCIException:
            return 0.0
    
    def get_traffic_light_state(self, tl_id: str) -> str:
        """
        Get current state of a traffic light.
        
        Args:
            tl_id: Traffic light ID
        
        Returns:
            str: Traffic light state string (e.g., "GGrrGGrr")
        """
        if not self._is_running:
            return ""
        
        try:
            return traci.trafficlight.getRedYellowGreenState(tl_id)
        except traci.TraCIException:
            return ""
    
    def set_traffic_light_state(self, tl_id: str, state: str) -> None:
        """
        Set traffic light state.
        
        Args:
            tl_id: Traffic light ID
            state: Traffic light state string (e.g., "GGrrGGrr")
        """
        if not self._is_running:
            logger.warning("Cannot set traffic light: SUMO not running")
            return
        
        try:
            traci.trafficlight.setRedYellowGreenState(tl_id, state)
        except traci.TraCIException as e:
            logger.error(f"Failed to set traffic light state: {e}")
    
    def get_vehicle_type(self, vehicle_id: str) -> str:
        """
        Get type of a specific vehicle.
        
        Args:
            vehicle_id: Vehicle ID
        
        Returns:
            str: Vehicle type (e.g., "car", "ambulance")
        """
        if not self._is_running:
            return ""
        
        try:
            return traci.vehicle.getTypeID(vehicle_id)
        except traci.TraCIException:
            return ""
    
    def get_all_vehicle_ids(self) -> List[str]:
        """
        Get IDs of all vehicles currently in simulation.
        
        Returns:
            list: List of vehicle IDs
        """
        if not self._is_running:
            return []
        
        try:
            return list(traci.vehicle.getIDList())
        except traci.TraCIException:
            return []
    
    def get_vehicle_route(self, vehicle_id: str) -> str:
        """
        Get the route ID of a vehicle.
        
        Args:
            vehicle_id: Vehicle ID
        
        Returns:
            str: Route ID
        """
        if not self._is_running:
            return ""
        
        try:
            return traci.vehicle.getRouteID(vehicle_id)
        except traci.TraCIException:
            return ""
    
    def is_running(self) -> bool:
        """
        Check if simulation is currently running.
        
        Returns:
            bool: True if simulation is active
        """
        return self._is_running and traci.isLoaded()
    
    def close(self) -> None:
        """
        Close SUMO simulation and cleanup resources.
        """
        if self._is_running:
            try:
                logger.debug(f"Closing SUMO - cumulative_arrived was: {self.cummulative_arrived}")
                # print(f"[WARNING] SUMO.close() called! cumulative={self.cummulative_arrived}")
                traci.close()
                self._is_running = False
                self._current_step = 0
                
            except Exception as e:
                logger.warning(f"Error during SUMO close: {e}")
        
        # Final cleanup
        self._route_file = None
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass  # Suppress exceptions during garbage collection


def kill_all_sumo_processes() -> int:
    """
    Kill all lingering SUMO processes system-wide.
    
    This is a safety measure to prevent resource leaks from crashed
    or orphaned SUMO processes.
    
    Returns:
        int: Number of processes killed
    """
    killed_count = 0
    
    try:
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                proc_name = proc.info.get('name', '').lower()
                if proc_name and 'sumo' in proc_name:
                    logger.debug(f"Killing SUMO process: {proc.info['pid']}")
                    proc.kill()
                    killed_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass  # Process already gone or no permission
        
        if killed_count > 0:
            logger.info(f"Killed {killed_count} lingering SUMO processes")
        
        return killed_count
        
    except Exception as e:
        logger.warning(f"SUMO process cleanup failed: {e}")
        return 0


if __name__ == "__main__":
    """Simple test of SUMO connection."""
    import tempfile
    from simulation.route_generator import RouteGenerator
    
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Testing SUMO Simulation Module")
    print("=" * 60)
    
    try:
        # Generate a test route file
        print("\n1. Generating test route file...")
        generator = RouteGenerator()
        route_file = generator.generate_unique_filename("test_sumo")
        generator.generate_route_file(route_file, scenario="WEEKEND", curriculum_stage=0)
        print(f"   ✓ Route file created: {route_file}")
        
        # Create SUMO simulation
        print("\n2. Initializing SUMO simulation...")
        sim = SUMOSimulation(use_gui=False)
        print(f"   ✓ SUMO manager created")
        
        # Start simulation
        print("\n3. Starting SUMO...")
        sim.start(route_file)
        print(f"   ✓ SUMO started successfully")
        
        # Run a few steps
        print("\n4. Running simulation steps...")
        for i in range(10):
            sim.step()
            veh_count = sim.get_vehicle_count()
            arrived = sim.get_arrived_vehicles()
            sim_time = sim.get_current_time()
            
            if i % 5 == 0:
                print(f"   Step {i}: time={sim_time}s, vehicles={veh_count}, arrived={arrived}")
        
        print(f"   ✓ Simulation steps successful")
        
        # Test data collection
        print("\n5. Testing data collection...")
        for direction, lanes in NetworkTopology.LANES.items():
            for lane in lanes:
                count = sim.get_lane_vehicle_count(lane)
                wait = sim.get_lane_waiting_time(lane)
                print(f"   {lane}: {count} vehicles, {wait:.1f}s wait time")
        
        # Close simulation
        print("\n6. Closing SUMO...")
        sim.close()
        print(f"   ✓ SUMO closed successfully")
        
        # Cleanup
        print("\n7. Cleaning up processes...")
        killed = kill_all_sumo_processes()
        print(f"   ✓ Cleaned up {killed} processes")
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)