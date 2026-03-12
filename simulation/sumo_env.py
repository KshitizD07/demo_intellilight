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

FIXED: Compatible with new configs/parameters.py structure
"""

import os
import sys
import time
import subprocess
import logging
from typing import Optional, List, Dict, Tuple
import psutil

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import traci
except ImportError:
    raise ImportError(
        "TraCI not found. Please install SUMO and ensure it's in your PATH. "
        "Visit: https://www.eclipse.org/sumo/"
    )

# Import from new parameters structure
from configs.parameters import simulation, paths

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

        # FIXED: Use new config structure
        self.sumo_cfg = sumo_cfg or simulation.SUMO_CONFIG_FILE
        self.use_gui = use_gui
        self.step_length = step_length or simulation.STEP_LENGTH
        self.timeout = timeout
        
        # Determine SUMO binary
        self.sumo_binary = "sumo-gui" if use_gui else "sumo"
        
        # Verify SUMO binary exists
        if not self._check_sumo_available():
            logger.warning(
                f"SUMO binary '{self.sumo_binary}' not found in PATH. "
                "Attempting to continue anyway..."
            )
        
        # Verify config file exists
        if not os.path.exists(self.sumo_cfg):
            logger.warning(
                f"SUMO config file not found: {self.sumo_cfg}\n"
                f"Will attempt to use when start() is called."
            )
        
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
            self.close()
        
        if not os.path.exists(route_file):
            raise FileNotFoundError(f"Route file not found: {route_file}")
        
        self._route_file = route_file
        
        # Build SUMO command
        sumo_cmd = self._build_command(route_file)
        logger.debug(f"Starting SUMO: {' '.join(sumo_cmd)}")
        
        try:
            # Clean up any zombie connections
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
            self._last_reported_arrivals = 0
            
            startup_time = time.time() - start_time
            logger.info(f"SUMO started successfully in {startup_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to start SUMO: {e}")
            
            # Ensure TraCI connection is cleared
            try:
                traci.close()
            except Exception:
                pass
                
            self._is_running = False
            raise RuntimeError(f"SUMO startup failed: {e}")
    
    def reset(self):
        """
        Reset the SUMO simulation.
        Safely closes existing simulation if running.
        """
        try:
            if traci.isLoaded():
                traci.close()
        except Exception:
            pass
        
        self._is_running = False
        self._current_step = 0
        self.cummulative_arrived = 0
        self._last_reported_arrivals = 0
    
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
            cmd.extend([
                "--no-step-log",
                "--no-warnings",
                "--duration-log.disable",
                "true"
            ])
        
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
            
            # Track arrived vehicles
            arrived_this_step = len(traci.simulation.getArrivedIDList())
            self.cummulative_arrived += arrived_this_step

            # Safety check: cumulative should never decrease
            if self.cummulative_arrived < self._last_reported_arrivals:
                logger.warning(
                    f"Cumulative counter went backwards! "
                    f"{self._last_reported_arrivals} -> {self.cummulative_arrived}"
                )
                self.cummulative_arrived = self._last_reported_arrivals
            
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
        
        # Update last reported for safety tracking
        self._last_reported_arrivals = self.cummulative_arrived
        
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
    
    def get_lane_halting_count(self, lane_id: str) -> int:
        """
        Get number of halting (stopped) vehicles on a lane.
        
        Args:
            lane_id: SUMO lane ID
        
        Returns:
            int: Number of halting vehicles
        """
        if not self._is_running:
            return 0
        
        try:
            return traci.lane.getLastStepHaltingNumber(lane_id)
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
    
    def get_lane_vehicle_ids(self, lane_id: str) -> List[str]:
        """
        Get IDs of all vehicles on a lane.
        
        Args:
            lane_id: SUMO lane ID
        
        Returns:
            list: List of vehicle IDs
        """
        if not self._is_running:
            return []
        
        try:
            return list(traci.lane.getLastStepVehicleIDs(lane_id))
        except traci.TraCIException:
            return []
    
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
    
    def get_vehicle_waiting_time(self, vehicle_id: str) -> float:
        """
        Get waiting time of a specific vehicle.
        
        Args:
            vehicle_id: Vehicle ID
        
        Returns:
            float: Waiting time in seconds
        """
        if not self._is_running:
            return 0.0
        
        try:
            return traci.vehicle.getWaitingTime(vehicle_id)
        except traci.TraCIException:
            return 0.0
    
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
        try:
            return self._is_running and traci.isLoaded()
        except Exception:
            return False
    
    def close(self) -> None:
        """
        Close SUMO simulation and cleanup resources.
        """
        try:
            # Only close if TraCI connection exists
            if traci.isLoaded():
                logger.debug(
                    f"Closing SUMO - cumulative_arrived was: {self.cummulative_arrived}"
                )
                traci.close()

        except Exception as e:
            logger.warning(f"Error during SUMO close: {e}")

        finally:
            self._is_running = False
            self._current_step = 0
            self._route_file = None
            self.cummulative_arrived = 0
            self._last_reported_arrivals = 0
    
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


# Simple standalone test
if __name__ == "__main__":
    """Simple test of SUMO connection."""
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print("TESTING SUMO SIMULATION MODULE")
    print("=" * 70)
    
    try:
        # Test 1: Check SUMO availability
        print("\n1. Checking SUMO availability...")
        sim = SUMOSimulation(use_gui=False)
        print("   ✅ SUMO manager created")
        
        # Test 2: Try to generate a route file and start
        print("\n2. Testing SUMO start...")
        
        try:
            from simulation.route_generator import RouteGenerator
            
            generator = RouteGenerator()
            route_file = generator.generate_unique_filename("test_sumo")
            generator.generate_route_file(
                route_file,
                scenario="WEEKEND",
                curriculum_stage=0
            )
            
            print(f"   Route file: {route_file}")
            
            sim.start(route_file)
            print("   ✅ SUMO started successfully")
            
            # Test 3: Run a few steps
            print("\n3. Running simulation steps...")
            for i in range(10):
                sim.step()
                if i % 5 == 0:
                    veh_count = sim.get_vehicle_count()
                    arrived = sim.get_arrived_vehicles()
                    time = sim.get_current_time()
                    print(f"   Step {i}: time={time}s, vehicles={veh_count}, arrived={arrived}")
            
            print("   ✅ Simulation steps successful")
            
            # Test 4: Close
            print("\n4. Closing SUMO...")
            sim.close()
            print("   ✅ SUMO closed successfully")
            
        except ImportError:
            print("   ⚠️  RouteGenerator not available, skipping start test")
        except Exception as e:
            print(f"   ⚠️  Start test failed: {e}")
        
        print("\n" + "=" * 70)
        print("✅ SUMO MODULE TEST COMPLETE")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()