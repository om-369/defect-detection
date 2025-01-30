"""Email notification system for critical events."""

import logging
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any
from typing import Dict
from typing import List

logger = logging.getLogger(__name__)


class EmailNotifier:
    """Email notification system."""

    def __init__(self):
        """Initialize email notifier with environment variables."""
        self.smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        self.sender_email = os.environ.get("NOTIFICATION_EMAIL")
        self.sender_password = os.environ.get("NOTIFICATION_PASSWORD")
        self.recipient_emails = os.environ.get("ALERT_RECIPIENTS", "").split(",")

    def is_configured(self) -> bool:
        """Check if email notification is properly configured."""
        return all(
            [
                self.smtp_server,
                self.smtp_port,
                self.sender_email,
                self.sender_password,
                self.recipient_emails,
            ]
        )

    def send_notification(
        self, subject: str, body: str, priority: str = "normal"
    ) -> bool:
        """Send email notification.

        Args:
            subject: Email subject
            body: Email body content
            priority: Priority level (low, normal, high)

        Returns:
            bool: True if email was sent successfully
        """
        if not self.is_configured():
            logger.warning("Email notification not configured")
            return False

        try:
            msg = MIMEMultipart()
            msg["From"] = self.sender_email
            msg["To"] = ", ".join(self.recipient_emails)
            msg["Subject"] = f"[Defect Detection] {subject}"

            # Add priority header
            if priority == "high":
                msg["X-Priority"] = "1"
            elif priority == "low":
                msg["X-Priority"] = "5"

            # Add timestamp and environment info to body
            full_body = f"""
            Timestamp: {datetime.utcnow().isoformat()}
            Environment: {os.environ.get('ENVIRONMENT', 'development')}
            Priority: {priority}
            
            {body}
            """

            msg.attach(MIMEText(full_body, "plain"))

            # Connect to SMTP server
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)

            logger.info(f"Notification sent: {subject}")
            return True

        except Exception as e:
            logger.error(f"Failed to send notification: {str(e)}")
            return False

    def notify_model_error(self, error: str) -> bool:
        """Send notification for model-related errors."""
        subject = "Model Error Detected"
        body = f"The following error occurred with the model:\n\n{error}"
        return self.send_notification(subject, body, priority="high")

    def notify_high_defect_rate(self, defect_rate: float, threshold: float) -> bool:
        """Send notification when defect rate exceeds threshold."""
        subject = "High Defect Rate Alert"
        body = f"""
        The current defect detection rate ({defect_rate:.2f}%) has exceeded
        the configured threshold ({threshold:.2f}%).
        
        Please investigate the production line for potential issues.
        """
        return self.send_notification(subject, body, priority="high")

    def notify_system_status(self, status: Dict[str, Any]) -> bool:
        """Send system status notification."""
        subject = f"System Status: {status['overall_status']}"
        body = f"""
        System Health Check Summary:
        
        Overall Status: {status['overall_status']}
        Healthy Checks: {status['healthy_checks']}/{status['total_checks']}
        
        Component Status:
        {self._format_component_status(status['components'])}
        
        Performance Metrics:
        - Average Response Time: {status['metrics']['avg_response_time']:.2f}ms
        - Error Rate: {status['metrics']['error_rate']:.2f}%
        - Memory Usage: {status['metrics']['memory_usage']:.1f}MB
        """
        return self.send_notification(subject, body, priority="normal")

    def _format_component_status(self, components: Dict[str, str]) -> str:
        """Format component status for email body."""
        return "\n".join(
            [f"- {component}: {status}" for component, status in components.items()]
        )


# Global notifier instance
notifier = EmailNotifier()
