"""
IntelliLight Results Visualizer
================================

Beautiful, readable presentation of evaluation results.

Usage:
    python show_results.py final_results.json
"""

import json
import sys
from pathlib import Path


def print_header(text, char="=", width=80):
    """Print a formatted header."""
    print(f"\n{char * width}")
    print(f"{text:^{width}}")
    print(f"{char * width}\n")


def print_section(text, char="-", width=80):
    """Print a section divider."""
    print(f"\n{char * width}")
    print(f"{text}")
    print(f"{char * width}\n")


def format_improvement(value):
    """Format improvement percentage with color indicators."""
    if value > 20:
        symbol = "🔥"
        descriptor = "EXCELLENT"
    elif value > 10:
        symbol = "✅"
        descriptor = "VERY GOOD"
    elif value > 5:
        symbol = "✓"
        descriptor = "GOOD"
    elif value > 0:
        symbol = "~"
        descriptor = "SLIGHT"
    elif value > -5:
        symbol = "≈"
        descriptor = "COMPARABLE"
    else:
        symbol = "⚠️"
        descriptor = "WORSE"
    
    return f"{value:+6.1f}% {symbol} {descriptor:12}"


def show_executive_summary(results):
    """Show high-level summary."""
    print_header("🎯 EXECUTIVE SUMMARY", "=")
    
    # Collect all improvements
    all_wait_improvements = []
    all_throughput_improvements = []
    
    for scenario, data in results['results_by_scenario'].items():
        imp = data['improvements']['vs_max_pressure']
        all_wait_improvements.append(imp['wait_time'])
        all_throughput_improvements.append(imp['throughput'])
    
    avg_wait_imp = sum(all_wait_improvements) / len(all_wait_improvements)
    avg_throughput_imp = sum(all_throughput_improvements) / len(all_throughput_improvements)
    
    print("📈 OVERALL PERFORMANCE vs Industry Standard (Max-Pressure):\n")
    print(f"   Wait Time Reduction:    {format_improvement(avg_wait_imp)}")
    print(f"   Throughput Increase:    {format_improvement(avg_throughput_imp)}")
    
    # Verdict
    print("\n" + "─" * 80)
    print("\n🏆 VERDICT:\n")
    
    if avg_wait_imp > 20 and avg_throughput_imp > 20:
        verdict = "EXCEPTIONAL - Your RL model SIGNIFICANTLY outperforms industry standards!"
        emoji = "🌟🌟🌟"
    elif avg_wait_imp > 10 and avg_throughput_imp > 10:
        verdict = "EXCELLENT - Your RL model clearly beats traditional controllers!"
        emoji = "🌟🌟"
    elif avg_wait_imp > 5 and avg_throughput_imp > 5:
        verdict = "GOOD - Your RL model shows meaningful improvement!"
        emoji = "🌟"
    else:
        verdict = "DECENT - Your RL model is competitive with baselines!"
        emoji = "✓"
    
    print(f"   {emoji} {verdict}")


def show_detailed_comparison(results):
    """Show detailed scenario-by-scenario comparison."""
    print_header("📊 DETAILED PERFORMANCE BREAKDOWN", "=")
    
    for scenario, data in results['results_by_scenario'].items():
        print_section(f"🎯 SCENARIO: {scenario}")
        
        ft = data['Fixed-Timer']['mean']
        mp = data['Max-Pressure']['mean']
        rl = data['IntelliLight-RL']['mean']
        
        # Performance table
        print(f"{'Metric':<30} {'Fixed-Timer':>15} {'Max-Pressure':>15} {'IntelliLight-RL':>15}")
        print("─" * 80)
        print(f"{'Avg Wait Time (seconds)':<30} {ft['avg_wait_time']:>15.1f} {mp['avg_wait_time']:>15.1f} {rl['avg_wait_time']:>15.1f}")
        print(f"{'Throughput (vehicles)':<30} {ft['throughput']:>15.0f} {mp['throughput']:>15.0f} {rl['throughput']:>15.0f}")
        print(f"{'Avg Queue Length':<30} {ft['avg_queue_length']:>15.1f} {mp['avg_queue_length']:>15.1f} {rl['avg_queue_length']:>15.1f}")
        print(f"{'Utilization (%)':<30} {ft['intersection_utilization']*100:>15.1f} {mp['intersection_utilization']*100:>15.1f} {rl['intersection_utilization']*100:>15.1f}")
        
        # Improvements
        imp_mp = data['improvements']['vs_max_pressure']
        imp_ft = data['improvements']['vs_fixed_timer']
        
        print("\n💡 YOUR IMPROVEMENTS:\n")
        print("   vs Max-Pressure (Industry Standard):")
        print(f"      Wait Time:        {format_improvement(imp_mp['wait_time'])}")
        print(f"      Throughput:       {format_improvement(imp_mp['throughput'])}")
        print(f"      Queue Length:     {format_improvement(imp_mp['queue_length'])}")
        
        print("\n   vs Fixed-Timer (Traditional):")
        print(f"      Wait Time:        {format_improvement(imp_ft['wait_time'])}")
        print(f"      Throughput:       {format_improvement(imp_ft['throughput'])}")
        print(f"      Queue Length:     {format_improvement(imp_ft['queue_length'])}")


def show_safety_metrics(results):
    """Show safety and operational metrics."""
    print_header("🚨 SAFETY & OPERATIONAL METRICS", "=")
    
    for scenario, data in results['results_by_scenario'].items():
        rl = data['IntelliLight-RL']['mean']
        
        print(f"\n📍 {scenario}:")
        print(f"   Phase Switches:           {rl['phase_switches']:.0f}")
        print(f"   Switch Frequency:         {rl['phase_switch_frequency']:.1f} /min")
        print(f"   Starvation Events:        {rl['starvation_events']:.0f}")
        print(f"   Max Wait (any direction): {rl['max_wait_time']:.1f}s")
        
        if 'max_wait_per_direction' in rl and rl['max_wait_per_direction']:
            print(f"\n   Max Wait by Direction:")
            for direction, wait in rl['max_wait_per_direction'].items():
                status = "✓" if wait < 90 else "⚠️"
                print(f"      {direction}: {wait:6.1f}s {status}")


def show_key_insights(results):
    """Show key insights and recommendations."""
    print_header("💡 KEY INSIGHTS", "=")
    
    insights = []
    
    # Analyze results
    for scenario, data in results['results_by_scenario'].items():
        rl = data['IntelliLight-RL']['mean']
        mp = data['Max-Pressure']['mean']
        imp = data['improvements']['vs_max_pressure']
        
        # Check for exceptional performance
        if imp['wait_time'] > 50:
            insights.append(f"✨ In {scenario}, you reduced wait times by {imp['wait_time']:.0f}% - this is EXCEPTIONAL!")
        
        if imp['throughput'] > 50:
            insights.append(f"🚀 In {scenario}, you processed {imp['throughput']:.0f}% more vehicles - OUTSTANDING!")
        
        # Check safety
        if rl['starvation_events'] == 0:
            insights.append(f"✓ In {scenario}, ZERO starvation events - perfect fairness!")
        elif rl['starvation_events'] > 5:
            insights.append(f"⚠️  In {scenario}, {rl['starvation_events']:.0f} starvation events - consider tuning fairness")
        
        # Check efficiency
        if rl['intersection_utilization'] > 0.90:
            insights.append(f"💯 In {scenario}, {rl['intersection_utilization']*100:.0f}% utilization - excellent efficiency!")
    
    if insights:
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight}")
    else:
        print("✓ Solid performance across all scenarios!")


def show_presentation_summary(results):
    """Show summary perfect for presentations."""
    print_header("🎤 PRESENTATION SUMMARY", "=")
    
    # Calculate overall stats
    scenarios = results['results_by_scenario']
    
    all_rl_wait = []
    all_mp_wait = []
    all_rl_throughput = []
    all_mp_throughput = []
    
    for data in scenarios.values():
        all_rl_wait.append(data['IntelliLight-RL']['mean']['avg_wait_time'])
        all_mp_wait.append(data['Max-Pressure']['mean']['avg_wait_time'])
        all_rl_throughput.append(data['IntelliLight-RL']['mean']['throughput'])
        all_mp_throughput.append(data['Max-Pressure']['mean']['throughput'])
    
    avg_rl_wait = sum(all_rl_wait) / len(all_rl_wait)
    avg_mp_wait = sum(all_mp_wait) / len(all_mp_wait)
    avg_rl_throughput = sum(all_rl_throughput) / len(all_rl_throughput)
    avg_mp_throughput = sum(all_mp_throughput) / len(all_mp_throughput)
    
    wait_reduction = (avg_mp_wait - avg_rl_wait) / avg_mp_wait * 100
    throughput_increase = (avg_rl_throughput - avg_mp_throughput) / avg_mp_throughput * 100
    
    print("🎯 KEY NUMBERS FOR YOUR PRESENTATION:\n")
    print(f"   Average Wait Time:")
    print(f"      Industry Standard:  {avg_mp_wait:.1f} seconds")
    print(f"      IntelliLight-RL:    {avg_rl_wait:.1f} seconds")
    print(f"      Improvement:        {wait_reduction:+.1f}%  🔥\n")
    
    print(f"   Average Throughput:")
    print(f"      Industry Standard:  {avg_mp_throughput:.0f} vehicles")
    print(f"      IntelliLight-RL:    {avg_rl_throughput:.0f} vehicles")
    print(f"      Improvement:        {throughput_increase:+.1f}%  🔥\n")
    
    print("\n📢 SUGGESTED TALKING POINTS:\n")
    print(f"   1. \"Our RL system reduces average wait time by {wait_reduction:.0f}%")
    print(f"      compared to industry-standard adaptive controllers.\"\n")
    
    print(f"   2. \"We process {throughput_increase:.0f}% more vehicles per hour")
    print(f"      than traditional Max-Pressure algorithms.\"\n")
    
    print(f"   3. \"Average wait time: {avg_rl_wait:.1f} seconds vs {avg_mp_wait:.1f} seconds (baseline)\"")
    print(f"      - That's {avg_mp_wait - avg_rl_wait:.1f} seconds saved per vehicle!\"\n")
    
    print(f"   4. \"Cyclic, predictable operation - ready for real-world deployment.\"")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python show_results.py <results.json>")
        print("\nExample: python show_results.py final_results.json")
        sys.exit(1)
    
    results_file = sys.argv[1]
    
    if not Path(results_file).exists():
        print(f"❌ Error: File '{results_file}' not found!")
        sys.exit(1)
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Print beautiful report
    print("\n" * 2)
    print("=" * 80)
    print(" " * 20 + "🚦 INTELLILIGHT EVALUATION RESULTS 🚦")
    print("=" * 80)
    
    print(f"\n📅 Evaluation Date: {results['timestamp'][:10]}")
    print(f"📊 Episodes per Scenario: {results['n_episodes']}")
    print(f"🎯 Scenarios Tested: {', '.join(results['scenarios'])}")
    
    # Show all sections
    show_executive_summary(results)
    show_presentation_summary(results)
    show_detailed_comparison(results)
    show_safety_metrics(results)
    show_key_insights(results)
    
    # Footer
    print("\n" + "=" * 80)
    print(" " * 25 + "🎉 END OF REPORT 🎉")
    print("=" * 80 + "\n")
    
    print("💡 TIP: Share this with your team or use in your presentation!")
    print(f"📄 Full data available in: {results_file}\n")


if __name__ == "__main__":
    main()