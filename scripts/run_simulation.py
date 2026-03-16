#!/usr/bin/env python3
"""CLI script to generate simulation data."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pipe_leak.config import SIM_CONFIG
from pipe_leak.simulation.network import build_pipe_network
from pipe_leak.simulation.events import generate_leak_events
from pipe_leak.utils.io import save_pipes, save_events


def main():
    print("=" * 60)
    print("Pipe Leak Simulation")
    print("=" * 60)

    print(f"\nConfig: {SIM_CONFIG.num_pipes} pipes, {SIM_CONFIG.simulation_years} years")
    print(f"Center: ({SIM_CONFIG.center_lat}, {SIM_CONFIG.center_lon})")

    # Generate network
    pipes_gdf = build_pipe_network(SIM_CONFIG)
    print(f"\nPipe network: {len(pipes_gdf)} pipes")
    print(f"Materials: {pipes_gdf['material'].value_counts().to_dict()}")
    print(f"Age range: {pipes_gdf['age'].min()}-{pipes_gdf['age'].max()} years")

    # Generate events
    events_df = generate_leak_events(pipes_gdf, SIM_CONFIG)
    if not events_df.empty:
        print(f"\nLeak events: {len(events_df)}")
        print(f"Severity: {events_df['severity'].value_counts().to_dict()}")
        print(f"Total repair cost: ${events_df['repair_cost'].sum():,.0f}")
        print(f"Unique pipes with leaks: {events_df['pipe_id'].nunique()}")
    else:
        print("\nNo leak events generated.")

    # Save
    save_pipes(pipes_gdf)
    save_events(events_df)
    print("\nData saved to data/processed/")


if __name__ == "__main__":
    main()
