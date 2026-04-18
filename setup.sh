#!/bin/bash
# c22os Telegram Bot setup script
# Run this after creating your Telegram bot via @BotFather

set -e

echo "═══════════════════════════════════════════"
echo "  c22os Telegram Bot Setup"
echo "═══════════════════════════════════════════"
echo ""

# Check for bot token
if grep -q "REPLACE_ME" .env; then
    echo "⚠️  You need to configure .env first:"
    echo ""
    echo "  1. Message @BotFather on Telegram → /newbot → get your token"
    echo "  2. Message @userinfobot on Telegram → get your user ID"
    echo "  3. Edit .env and replace REPLACE_ME values:"
    echo "     - TELEGRAM_BOT_TOKEN=your_token_here"
    echo "     - ALLOWED_USERS=your_user_id"
    echo "     - NOTIFICATION_CHAT_IDS=your_user_id"
    echo ""
    echo "Then re-run this script."
    exit 1
fi

# Extract chat ID from .env
CHAT_ID=$(grep "^NOTIFICATION_CHAT_IDS=" .env | cut -d= -f2)
if [ -z "$CHAT_ID" ]; then
    echo "❌ NOTIFICATION_CHAT_IDS not set in .env"
    exit 1
fi

echo "✓ .env configured (chat_id=$CHAT_ID)"

# Ensure data directory
mkdir -p data

# Seed cron jobs
echo ""
echo "Seeding scheduled jobs..."
.venv/bin/python seed_jobs.py --chat-id "$CHAT_ID"

# Set timezone
echo ""
echo "✓ Timezone: America/New_York (EST/EDT)"
echo "  All cron times are in this timezone."

# Install systemd service
echo ""
echo "Installing systemd user service..."
mkdir -p ~/.config/systemd/user
cp c22os-telegram.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable c22os-telegram.service
echo "✓ Service installed and enabled"

echo ""
echo "═══════════════════════════════════════════"
echo "  Setup Complete!"
echo "═══════════════════════════════════════════"
echo ""
echo "Commands:"
echo "  Start:   systemctl --user start c22os-telegram"
echo "  Stop:    systemctl --user stop c22os-telegram"
echo "  Status:  systemctl --user status c22os-telegram"
echo "  Logs:    journalctl --user -u c22os-telegram -f"
echo ""
echo "Or run directly:"
echo "  cd $(pwd) && .venv/bin/claude-telegram-bot"
echo ""
echo "Test by messaging your bot on Telegram!"
