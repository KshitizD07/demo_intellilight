"""
Realistic Traffic Generator for IntelliLight
=============================================

Generates traffic patterns that match real-world behavior:
- Time-of-day demand curves
- Random events (accidents, weather, special events)
- Directional imbalances
- High volatility (±30% variance)

This replaces the simple uniform traffic generation.
"""

import random
import numpy as np
from enum import Enum
from typing import Dict, List, Tuple
from dataclasses import dataclass


class TrafficEvent(Enum):
    """Random events that affect traffic."""
    NORMAL = "normal"
    ACCIDENT = "accident"
    RAIN = "rain"
    SPECIAL_EVENT = "special_event"
    ROAD_WORK = "road_work"


@dataclass
class TrafficDemand:
    """Traffic demand for all directions."""
    north: int
    south: int
    east: int
    west: int
    
    @property
    def total(self) -> int:
        return self.north + self.south + self.east + self.west
    
    def to_dict(self) -> Dict[str, int]:
        return {
            'N': self.north,
            'S': self.south,
            'E': self.east,
            'W': self.west
        }


class RealisticTrafficGenerator:
    """
    Generates realistic traffic patterns for simulation.
    
    Features:
    - Time-of-day curves (morning rush, evening rush, weekend)
    - Random events that create spikes
    - Directional imbalances (e.g., morning: suburbs→city)
    - High variance (realistic uncertainty)
    """
    
    def __init__(
        self,
        base_demand: int = 800,
        volatility: float = 0.3,
        event_probability: float = 0.10,
        enable_time_curves: bool = True,
        seed: int = None
    ):
        """
        Initialize traffic generator.
        
        Args:
            base_demand: Base vehicles per hour
            volatility: Random variance (0.3 = ±30%)
            event_probability: Chance of random event per episode
            enable_time_curves: Use time-of-day patterns
            seed: Random seed for reproducibility
        """
        self.base_demand = base_demand
        self.volatility = volatility
        self.event_probability = event_probability
        self.enable_time_curves = enable_time_curves
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Current state
        self.current_event = TrafficEvent.NORMAL
        self.event_duration = 0
        self.time_of_day = 8.0  # Start at 8 AM

        self.turn_ratios = {
            "straight": 0.6,
            "left": 0.25,
            "right": 0.15
        }
    
    def generate_demand(
        self,
        time_of_day: float = None,
        force_event: TrafficEvent = None
    ) -> TrafficDemand:
        """
        Generate traffic demand for current time.
        
        Args:
            time_of_day: Hour of day (0-24), None = use internal time
            force_event: Force specific event (for testing)
        
        Returns:
            TrafficDemand with vehicles/hour for each direction
        """
        if time_of_day is not None:
            self.time_of_day = time_of_day
        
        # 1. Get base demand from time-of-day curve
        # if self.enable_time_curves:
        #     base = self._apply_time_curve(self.base_demand, self.time_of_day)
        # else:
        #     base = self.base_demand

        difficulty_scale = 0.6 + random.random() * 0.4
        base = int(self._apply_time_curve(self.base_demand, self.time_of_day) * difficulty_scale)
        
        # 2. Apply random events
        if force_event:
            self.current_event = force_event
            self.event_duration = random.randint(3, 10)
        elif self.event_duration > 0:
            self.event_duration -= 1
        elif random.random() < self.event_probability:
            self._trigger_random_event()
        else:
            self.current_event = TrafficEvent.NORMAL
        
        # 3. Apply event effects
        base = self._apply_event_effects(base)
        
        # 4. Add directional imbalances
        demands = self._apply_directional_patterns(base, self.time_of_day)
        
        # 5. Add random noise (volatility)
        demands = self._add_volatility(demands)

        demands = self._apply_turning_movements(demands)
        
        # 6. Ensure minimum demand (at least 1 vehicle)
        demands = {k: max(1, v) for k, v in demands.items()}
        
        return TrafficDemand(
            north=int(demands['N']),
            south=int(demands['S']),
            east=int(demands['E']),
            west=int(demands['W'])
        )
    
    def _apply_time_curve(self, base_demand: int, time: float) -> int:
        """
        Apply time-of-day demand curve.
        
        Typical urban pattern:
        - 6-9 AM: Morning rush (high inbound to city)
        - 9 AM-4 PM: Moderate midday traffic
        - 4-7 PM: Evening rush (high outbound from city)
        - 7 PM-6 AM: Light overnight traffic
        """
        hour = time % 24
        
        # Morning rush (7-9 AM): 1.5x peak
        if 7 <= hour < 9:
            multiplier = 1.0 + 0.5 * np.exp(-((hour - 8) ** 2) / 0.5)
        
        # Evening rush (5-7 PM): 1.4x peak
        elif 17 <= hour < 19:
            multiplier = 1.0 + 0.4 * np.exp(-((hour - 18) ** 2) / 0.5)
        
        # Midday (9 AM - 5 PM): 0.8-1.0x
        elif 9 <= hour < 17:
            multiplier = 0.8 + 0.2 * (1 + np.sin((hour - 9) * np.pi / 8)) / 2
        
        # Evening/Night (7 PM - 7 AM): 0.3-0.6x
        else:
            if hour >= 19:
                night_hour = hour - 19
            else:
                night_hour = hour + 5
            multiplier = 0.3 + 0.3 * (1 + np.cos(night_hour * np.pi / 12)) / 2
        
        return int(base_demand * multiplier)
    
    def _trigger_random_event(self):
        """Trigger a random traffic event."""
        events = list(TrafficEvent)
        events.remove(TrafficEvent.NORMAL)
        
        # Weighted probabilities
        weights = [0.5, 0.25, 0.15, 0.10]  # Accident most common
        
        self.current_event = random.choices(events, weights=weights)[0]
        self.event_duration = random.randint(3, 10)  # Lasts 3-10 episodes
    
    def _apply_event_effects(self, base_demand: int) -> int:
        """
        Apply effects of current event.
        
        Events affect total demand:
        - Accident: +50% (rubbernecking, congestion spillover)
        - Rain: +30% (slower speeds, more cautious driving)
        - Special Event: +100% (sports game, concert)
        - Road Work: -20% (some avoid area)
        """
        if self.current_event == TrafficEvent.ACCIDENT:
            return int(base_demand * 1.5)
        elif self.current_event == TrafficEvent.RAIN:
            return int(base_demand * 1.3)
        elif self.current_event == TrafficEvent.SPECIAL_EVENT:
            return int(base_demand * 2.0)
        elif self.current_event == TrafficEvent.ROAD_WORK:
            return int(base_demand * 0.8)
        else:
            return base_demand
    
    def _apply_directional_patterns(
        self,
        total_demand: int,
        time: float
    ) -> Dict[str, float]:
        """
        Split total demand across directions based on time.
        
        Morning (7-9 AM): Heavy inbound (N, W → city center)
        Evening (5-7 PM): Heavy outbound (city → S, E)
        Midday/Night: Balanced
        """
        hour = time % 24
        
        # Morning rush: Inbound bias (N=35%, S=15%, E=15%, W=35%)
        if 7 <= hour < 9:
            split = {'N': 0.35, 'S': 0.15, 'E': 0.15, 'W': 0.35}
        
        # Evening rush: Outbound bias (N=15%, S=35%, E=35%, W=15%)
        elif 17 <= hour < 19:
            split = {'N': 0.15, 'S': 0.35, 'E': 0.35, 'W': 0.15}
        
        # Balanced (all equal)
        else:
            split = {'N': 0.25, 'S': 0.25, 'E': 0.25, 'W': 0.25}
        
        # Apply event-specific directional biases
        if self.current_event == TrafficEvent.ACCIDENT:
            # Accident on one direction: spike that direction
            spike_dir = random.choice(['N', 'S', 'E', 'W'])
            split[spike_dir] *= 2.0
            # Normalize
            total = sum(split.values())
            split = {k: v / total for k, v in split.items()}
        
        elif self.current_event == TrafficEvent.SPECIAL_EVENT:
            # Event at city center: all directions heavy inbound
            split = {'N': 0.3, 'S': 0.3, 'E': 0.2, 'W': 0.2}
        
        # Calculate actual demands
        return {k: total_demand * v for k, v in split.items()}
    
    def _add_volatility(self, demands: Dict[str, float]) -> Dict[str, float]:
        """
        Add random variance to each direction.
        
        Real traffic has high uncertainty:
        - Individual vehicle arrivals are random
        - Driver behavior varies
        - Route choices change
        
        We add ±30% noise to each direction independently.
        """
        noisy_demands = {}
        for direction, demand in demands.items():
            # Random multiplier: 0.7 to 1.3 (for 30% volatility)
            noise = random.uniform(1 - self.volatility, 1 + self.volatility)
            noisy_demands[direction] = demand * noise
        
        return noisy_demands
    
    def _apply_turning_movements(self, demands: Dict[str, float]) -> Dict[str, float]:
    

        adjusted = {"N": 0, "S": 0, "E": 0, "W": 0}

        for direction, demand in demands.items():

            straight = demand * self.turn_ratios["straight"]
            left = demand * self.turn_ratios["left"]
            right = demand * self.turn_ratios["right"]

            if direction == "N":
                adjusted["S"] += straight
                adjusted["E"] += left
                adjusted["W"] += right

            elif direction == "S":
                adjusted["N"] += straight
                adjusted["W"] += left
                adjusted["E"] += right

            elif direction == "E":
                adjusted["W"] += straight
                adjusted["N"] += left
                adjusted["S"] += right

            elif direction == "W":
                adjusted["E"] += straight
                adjusted["S"] += left
                adjusted["N"] += right

        return adjusted
    
    def get_scenario_demand(self, scenario: str) -> TrafficDemand:
        """
        Get predefined scenario demands (for testing).
        
        Args:
            scenario: One of MORNING_RUSH, EVENING_RUSH, WEEKEND, NIGHT, ACCIDENT
        
        Returns:
            TrafficDemand for that scenario
        """
        if scenario == "MORNING_RUSH":
            return self.generate_demand(time_of_day=8.0)
        
        elif scenario == "EVENING_RUSH":
            return self.generate_demand(time_of_day=18.0)
        
        elif scenario == "WEEKEND":
            # Weekend: lower demand, balanced directions
            weekend_demand = int(self.base_demand * 0.6)
            split = weekend_demand // 4
            return TrafficDemand(split, split, split, split)
        
        elif scenario == "NIGHT":
            return self.generate_demand(time_of_day=2.0)
        
        elif scenario == "ACCIDENT":
            return self.generate_demand(
                time_of_day=8.0,
                force_event=TrafficEvent.ACCIDENT
            )
        
        else:
            # Default: balanced moderate traffic
            split = self.base_demand // 4
            return TrafficDemand(split, split, split, split)
    
    def reset(self):
        """Reset to initial state."""
        self.current_event = TrafficEvent.NORMAL
        self.event_duration = 0
        self.time_of_day = 8.0
    
    def get_event_info(self) -> Dict:
        """Get current event information."""
        return {
            'event': self.current_event.value,
            'duration_remaining': self.event_duration,
            'time_of_day': self.time_of_day
        }


def test_traffic_generator():
    """Test the traffic generator with various scenarios."""
    print("=" * 70)
    print("REALISTIC TRAFFIC GENERATOR TEST")
    print("=" * 70)
    
    generator = RealisticTrafficGenerator(
        base_demand=800,
        volatility=0.3,
        event_probability=0.0,  # Disable random events for testing
        enable_time_curves=True
    )
    
    # Test time-of-day curves
    print("\n📊 TIME-OF-DAY DEMAND CURVES:")
    print(f"\n{'Time':<12} {'Total Demand':<15} {'N':<8} {'S':<8} {'E':<8} {'W':<8}")
    print("-" * 70)
    
    test_times = [2, 6, 8, 12, 18, 22]  # Night, Early, Morning, Noon, Evening, Late
    
    for hour in test_times:
        demand = generator.generate_demand(time_of_day=hour)
        time_str = f"{int(hour):02d}:00"
        print(f"{time_str:<12} {demand.total:<15} {demand.north:<8} {demand.south:<8} {demand.east:<8} {demand.west:<8}")
    
    # Test events
    print("\n\n🚨 EVENT EFFECTS:")
    print(f"\n{'Event':<20} {'Total Demand':<15} {'N':<8} {'S':<8} {'E':<8} {'W':<8}")
    print("-" * 70)
    
    events = [
        TrafficEvent.NORMAL,
        TrafficEvent.ACCIDENT,
        TrafficEvent.RAIN,
        TrafficEvent.SPECIAL_EVENT,
        TrafficEvent.ROAD_WORK
    ]
    
    for event in events:
        demand = generator.generate_demand(time_of_day=8.0, force_event=event)
        print(f"{event.value:<20} {demand.total:<15} {demand.north:<8} {demand.south:<8} {demand.east:<8} {demand.west:<8}")
    
    # Test volatility
    print("\n\n📈 VOLATILITY TEST (10 samples at 8 AM):")
    print(f"\n{'Sample':<10} {'Total Demand':<15} {'N':<8} {'S':<8} {'E':<8} {'W':<8}")
    print("-" * 70)
    
    generator.reset()
    for i in range(10):
        demand = generator.generate_demand(time_of_day=8.0)
        print(f"{i+1:<10} {demand.total:<15} {demand.north:<8} {demand.south:<8} {demand.east:<8} {demand.west:<8}")
    
    # Test predefined scenarios
    print("\n\n🎯 PREDEFINED SCENARIOS:")
    print(f"\n{'Scenario':<20} {'Total Demand':<15} {'N':<8} {'S':<8} {'E':<8} {'W':<8}")
    print("-" * 70)
    
    scenarios = ["MORNING_RUSH", "EVENING_RUSH", "WEEKEND", "NIGHT", "ACCIDENT"]
    
    generator.reset()
    for scenario in scenarios:
        demand = generator.get_scenario_demand(scenario)
        print(f"{scenario:<20} {demand.total:<15} {demand.north:<8} {demand.south:<8} {demand.east:<8} {demand.west:<8}")
    
    print("\n" + "=" * 70)
    print("✅ TEST COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_traffic_generator()