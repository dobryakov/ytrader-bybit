#!/usr/bin/env python3
"""Script to analyze subscriptions and identify stale/unused ones.

Usage:
    docker compose exec ws-gateway python /app/scripts/analyze_subscriptions.py
"""

import asyncio
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from services.database.connection import DatabaseConnection
from services.database.subscription_repository import SubscriptionRepository
from config.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

# Thresholds
WARNING_MINUTES = 5
CRITICAL_MINUTES = 30
AUTO_DEACTIVATE_MINUTES = 60


async def analyze_subscriptions():
    """Analyze all subscriptions and report stale ones."""
    await DatabaseConnection.create_pool()
    
    try:
        all_subscriptions = await SubscriptionRepository.list_subscriptions()
        now = datetime.now(timezone.utc)
        
        active_subscriptions = [s for s in all_subscriptions if s.is_active]
        inactive_subscriptions = [s for s in all_subscriptions if not s.is_active]
        
        print(f"\n=== Subscription Analysis ===")
        print(f"Total subscriptions: {len(all_subscriptions)}")
        print(f"Active: {len(active_subscriptions)}")
        print(f"Inactive: {len(inactive_subscriptions)}")
        
        # Analyze active subscriptions
        stale_warning = []
        stale_critical = []
        stale_auto_deactivate = []
        never_received = []
        healthy = []
        
        for subscription in active_subscriptions:
            if not subscription.last_event_at:
                # Check if created long ago
                if subscription.created_at:
                    created_at = subscription.created_at
                    if created_at.tzinfo is None:
                        created_at = created_at.replace(tzinfo=timezone.utc)
                    else:
                        created_at = created_at.astimezone(timezone.utc)
                    
                    age_since_creation = now - created_at
                    if age_since_creation > timedelta(minutes=WARNING_MINUTES):
                        never_received.append((subscription, age_since_creation))
                else:
                    never_received.append((subscription, None))
                continue
            
            # Normalize last_event_at
            last_event_at = subscription.last_event_at
            if last_event_at.tzinfo is None:
                last_event_at = last_event_at.replace(tzinfo=timezone.utc)
            else:
                last_event_at = last_event_at.astimezone(timezone.utc)
            
            age = now - last_event_at
            age_minutes = age.total_seconds() / 60
            
            if age > timedelta(minutes=AUTO_DEACTIVATE_MINUTES):
                stale_auto_deactivate.append((subscription, age))
            elif age > timedelta(minutes=CRITICAL_MINUTES):
                stale_critical.append((subscription, age))
            elif age > timedelta(minutes=WARNING_MINUTES):
                stale_warning.append((subscription, age))
            else:
                healthy.append((subscription, age))
        
        # Print results
        print(f"\n=== Active Subscriptions Status ===")
        print(f"Healthy: {len(healthy)}")
        print(f"Warning (>{WARNING_MINUTES}m stale): {len(stale_warning)}")
        print(f"Critical (>{CRITICAL_MINUTES}m stale): {len(stale_critical)}")
        print(f"Auto-deactivate (>{AUTO_DEACTIVATE_MINUTES}m stale): {len(stale_auto_deactivate)}")
        print(f"Never received events: {len(never_received)}")
        
        # Details
        if stale_warning:
            print(f"\n--- Warning Subscriptions (>{WARNING_MINUTES}m) ---")
            for subscription, age in stale_warning:
                print(f"  {subscription.id}")
                print(f"    Service: {subscription.requesting_service}")
                print(f"    Channel: {subscription.channel_type} | Topic: {subscription.topic}")
                print(f"    Last event: {subscription.last_event_at} ({age.total_seconds() / 60:.1f}m ago)")
        
        if stale_critical:
            print(f"\n--- Critical Subscriptions (>{CRITICAL_MINUTES}m) ---")
            for subscription, age in stale_critical:
                print(f"  {subscription.id}")
                print(f"    Service: {subscription.requesting_service}")
                print(f"    Channel: {subscription.channel_type} | Topic: {subscription.topic}")
                print(f"    Last event: {subscription.last_event_at} ({age.total_seconds() / 60:.1f}m ago)")
        
        if stale_auto_deactivate:
            print(f"\n--- Auto-Deactivate Subscriptions (>{AUTO_DEACTIVATE_MINUTES}m) ---")
            for subscription, age in stale_auto_deactivate:
                print(f"  {subscription.id}")
                print(f"    Service: {subscription.requesting_service}")
                print(f"    Channel: {subscription.channel_type} | Topic: {subscription.topic}")
                print(f"    Last event: {subscription.last_event_at} ({age.total_seconds() / 60:.1f}m ago)")
        
        if never_received:
            print(f"\n--- Never Received Events ---")
            for subscription, age_since_creation in never_received:
                print(f"  {subscription.id}")
                print(f"    Service: {subscription.requesting_service}")
                print(f"    Channel: {subscription.channel_type} | Topic: {subscription.topic}")
                if age_since_creation:
                    print(f"    Created: {subscription.created_at} ({age_since_creation.total_seconds() / 60:.1f}m ago)")
                else:
                    print(f"    Created: {subscription.created_at}")
        
        # Group by service
        print(f"\n=== By Service ===")
        by_service = {}
        for subscription in active_subscriptions:
            service = subscription.requesting_service
            if service not in by_service:
                by_service[service] = []
            by_service[service].append(subscription)
        
        for service, subs in sorted(by_service.items()):
            print(f"\n{service}: {len(subs)} subscriptions")
            for sub in subs:
                status = "OK"
                if not sub.last_event_at:
                    status = "NO_EVENTS"
                else:
                    last_event_at = sub.last_event_at
                    if last_event_at.tzinfo is None:
                        last_event_at = last_event_at.replace(tzinfo=timezone.utc)
                    else:
                        last_event_at = last_event_at.astimezone(timezone.utc)
                    age = now - last_event_at
                    age_minutes = age.total_seconds() / 60
                    if age_minutes > AUTO_DEACTIVATE_MINUTES:
                        status = f"STALE_{age_minutes:.0f}m"
                    elif age_minutes > CRITICAL_MINUTES:
                        status = f"CRITICAL_{age_minutes:.0f}m"
                    elif age_minutes > WARNING_MINUTES:
                        status = f"WARNING_{age_minutes:.0f}m"
                
                print(f"  - {sub.channel_type} ({sub.topic}): {status}")
        
        # Recommendations
        print(f"\n=== Recommendations ===")
        if stale_auto_deactivate:
            print(f"⚠️  {len(stale_auto_deactivate)} subscriptions should be auto-deactivated (>{AUTO_DEACTIVATE_MINUTES}m stale)")
            print("   These will be automatically deactivated by subscription_monitor")
        
        if stale_critical:
            print(f"⚠️  {len(stale_critical)} subscriptions are critical (>{CRITICAL_MINUTES}m stale)")
            print("   Check WebSocket connection and Bybit API status")
        
        if never_received:
            print(f"⚠️  {len(never_received)} subscriptions never received events")
            print("   Check if subscriptions are correctly sent to WebSocket")
        
        if inactive_subscriptions:
            print(f"\n💡 {len(inactive_subscriptions)} inactive subscriptions can be cleaned up")
            print("   Consider running cleanup script to remove old inactive subscriptions")
        
    finally:
        await DatabaseConnection.close_pool()


if __name__ == "__main__":
    asyncio.run(analyze_subscriptions())

