"""Seed data for the mock ticketing API. Fictional people and systems."""
from datetime import datetime, timedelta

_BASE = datetime(2026, 9, 1, 9, 0, 0)

def _t(n, title, description, status, priority, category, requester, assignee=None, days_ago=0, comments=()):
    created = _BASE + timedelta(days=n % 23, hours=n % 7)
    return {
        "id": f"TCK-{1000 + n}",
        "title": title,
        "description": description,
        "status": status,
        "priority": priority,
        "category": category,
        "requester": requester,
        "assignee": assignee,
        "created_at": created.isoformat() + "Z",
        "updated_at": (created + timedelta(hours=3 + n % 11)).isoformat() + "Z",
        "comments": [
            {"id": f"CMT-{1000 + n}-{i + 1}", "author": a, "body": b, "created_at": (created + timedelta(hours=i + 1)).isoformat() + "Z"}
            for i, (a, b) in enumerate(comments)
        ],
    }

SEED = [
    _t(1, "VPN disconnects every 10 minutes", "Working from home, the VPN client drops roughly every ten minutes and reconnects. Started after the Monday update.", "open", "high", "network", "Priya Raman"),
    _t(2, "Cannot reset password via self-service portal", "The reset link in the email expires before I can click it.", "in_progress", "medium", "access", "Tom Okafor", "Dana Whitfield", comments=[("Dana Whitfield", "Reproduced. Link TTL is 2 minutes on the staging mail relay; checking prod.")]),
    _t(3, "Laptop replacement request", "Current laptop is 5 years old, fan is loud and battery lasts 40 minutes.", "open", "low", "hardware", "Lena Fischer"),
    _t(4, "Need access to the analytics dashboard", "Joined the data team this week; need viewer access to the analytics dashboards.", "resolved", "medium", "access", "Marcus Bell", "Dana Whitfield", comments=[("Dana Whitfield", "Access granted, viewer role."), ("Marcus Bell", "Confirmed, thanks.")]),
    _t(5, "Printer on floor 3 keeps jamming", "The shared printer near the kitchen jams on every duplex job.", "open", "low", "hardware", "Aisha Khan"),
    _t(6, "Outlook calendar not syncing to phone", "Calendar on the phone is a day behind. Mail works.", "in_progress", "medium", "software", "Jonas Berg", "Ravi Nair"),
    _t(7, "VPN access for contractor", "Contractor starting next week needs VPN access for 3 months.", "open", "medium", "access", "Priya Raman"),
    _t(8, "Software license for design tool", "Need a seat on the design tool for the new hire in marketing.", "open", "low", "software", "Sofia Marin"),
    _t(9, "Two-factor device lost", "Lost my phone over the weekend; cannot approve 2FA prompts.", "resolved", "urgent", "access", "Tom Okafor", "Ravi Nair", comments=[("Ravi Nair", "Temporary bypass issued for 24h, new device enrolled.")]),
    _t(10, "Wi-Fi weak in meeting room B", "Calls drop in meeting room B; signal shows one bar.", "open", "medium", "network", "Marcus Bell"),
    _t(11, "Monitor flickers when docked", "External monitor flickers when the laptop is in the dock; fine on direct cable.", "open", "low", "hardware", "Lena Fischer"),
    _t(12, "Shared drive permission denied", "Getting permission denied on the finance shared drive since this morning.", "in_progress", "high", "access", "Aisha Khan", "Dana Whitfield"),
    _t(13, "Request for second monitor", "Would like a second monitor for the home office.", "closed", "low", "hardware", "Jonas Berg", "Ravi Nair", comments=[("Ravi Nair", "Shipped."), ("Jonas Berg", "Received.")]),
    _t(14, "Email flagged as spam by customers", "Customers report our emails land in spam since last week.", "open", "high", "software", "Sofia Marin"),
    _t(15, "Keyboard keys sticking", "Several keys on the laptop keyboard stick.", "open", "low", "hardware", "Priya Raman"),
    _t(16, "Cannot install approved package", "Package manager blocked by endpoint protection for an approved package.", "in_progress", "medium", "software", "Marcus Bell", "Ravi Nair"),
    _t(17, "VPN split tunnelling question", "Can we enable split tunnelling for video calls?", "open", "low", "network", "Tom Okafor"),
    _t(18, "Onboarding accounts for new starter", "New starter on the 14th needs email, chat and HR system accounts.", "open", "medium", "access", "Sofia Marin"),
    _t(19, "Phone extension not ringing", "Desk phone extension goes straight to voicemail.", "resolved", "low", "network", "Aisha Khan", "Dana Whitfield"),
    _t(20, "Build server disk full", "CI builds failing with no space left on device.", "in_progress", "urgent", "infrastructure", "Jonas Berg", "Ravi Nair", comments=[("Ravi Nair", "Cleared 40GB of old artifacts; adding retention policy.")]),
    _t(21, "Password expiry too frequent", "Password expires every 30 days; policy says 90.", "open", "low", "access", "Lena Fischer"),
    _t(22, "Projector in boardroom not detected", "Laptop does not detect the boardroom projector over HDMI.", "open", "medium", "hardware", "Marcus Bell"),
    _t(23, "Antivirus slowing laptop", "Full scans run during working hours and freeze the machine.", "open", "medium", "software", "Priya Raman"),
    _t(24, "Need admin rights for local dev", "Need local admin to run containers for development.", "in_progress", "medium", "access", "Tom Okafor", "Dana Whitfield"),
]
