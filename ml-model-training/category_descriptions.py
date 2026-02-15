# Category-specific description templates for realistic ITSM tickets
# These help ML models learn distinguishing features for each category

CATEGORY_DESCRIPTIONS = {
    "Network": [
        "VPN tunnel keeps dropping every few minutes. Users cannot access internal resources. Need immediate fix.",
        "Router in building C is not responding. Network connectivity lost for 50+ users. Critical issue.",
        "DNS server not resolving domain names. Intermittent connection to internet and intranet sites.",
        "Firewall rules blocking access to cloud applications. Users getting timeout errors.",
        "WiFi signal strength very weak on 3rd floor. Constant disconnections reported by multiple employees.",
        "Switch port in conference room not functioning. Cannot connect ethernet cable for presentations.",
        "Network drive mapping fails with error 'path not found'. Users unable to access shared folders.",
        "DHCP not assigning IP addresses. Clients showing 169.254.x.x addresses and no connectivity.",
        "Cannot ping production server from workstation. Network latency very high, packet loss 40%.",
        "Load balancer not distributing traffic. All requests going to single server causing performance issues.",
    ],
    "Hardware": [
        "Laptop screen flickering and showing horizontal lines. Display becomes unreadable after 10 minutes of use.",
        "Desktop computer won't power on at all. No lights, no beep codes, completely dead. Need replacement urgently.",
        "External monitor has black screen. Checked cables and power, still not displaying anything.",
        "Wireless keyboard not responding to keystrokes. Changed batteries, still not working properly.",
        "Mouse cursor moving erratically across screen. Cleaned mouse but issue persists.",
        "Office printer showing paper jam error but no paper visible. Already tried turning off and on.",
        "Hard drive making clicking noise. Computer very slow and files not opening. Suspect drive failure.",
        "Docking station not detecting laptop. Monitor and peripherals not connecting through dock.",
        "Laptop battery drains in 30 minutes even when fully charged. Battery health showing critical.",
        "Desktop computer overheating and shutting down randomly. Fans running very loud and hot air blowing.",
    ],
    "Software": [
        "Microsoft Outlook crashes immediately on startup. Error message 'Outlook.exe stopped working' appears.",
        "Software update for SAP ERP failed halfway. Now system won't load and showing database error.",
        "Cannot connect to SQL Server database. Application timing out with 'connection refused' error.",
        "Email client not syncing with Exchange server. Emails from yesterday still not downloaded.",
        "CRM application extremely slow. Takes 5+ minutes to load customer records. Unusable.",
        "ERP system shows timeout error when running monthly reports. Query execution exceeds 30 seconds.",
        "File sharing application gives permission denied error. Cannot upload documents to team folder.",
        "Software license expired for Adobe Creative Suite. Users unable to launch Photoshop and Illustrator.",
        "Cannot install latest version of accounting software. Setup fails with error code 0x80070643.",
        "Program freezes when opening large Excel files. CPU usage spikes to 100% and system becomes unresponsive.",
    ],
    "Access": [
        "Cannot login to employee portal. Username and password correct but getting 'invalid credentials' error.",
        "Password reset link in email not working. Click takes me to blank page instead of reset form.",
        "User account locked out after 3 failed login attempts. Need immediate unlock as user needs urgent access.",
        "Access denied when trying to open HR shared folder. Getting error 'insufficient permissions'.",
        "User permissions not updated correctly. Recent promotion to manager but still have read-only access.",
        "MFA authentication token not working. Authenticator app showing different code than system expects.",
        "Need access to finance database for quarterly reporting. Require read access to all tables.",
        "Unable to access Salesforce application. Account active but getting unauthorized access message.",
        "Account disabled without notice. User received email saying account deactivated but needs access immediately.",
        "Single sign-on authentication failing for cloud applications. Getting redirect loop error on login page.",
    ]
}

def get_category_description(category: str) -> str:
    """Return a random description for the given category."""
    import random
    return random.choice(CATEGORY_DESCRIPTIONS.get(category, ["General IT issue requiring attention."]))
