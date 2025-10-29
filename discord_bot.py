"""
Discord bot for RegSHO Threshold reporting using Pycord.

This version exposes a "/regsho_tracker list" slash command and posts
daily reports at 3 AM Eastern.  The bot must be invited with
the applications.commands scope for the slash command to appear.
"""

import asyncio
import os
from datetime import datetime, timedelta, time as dt_time
from typing import Optional
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    ZoneInfo = None

import discord

async def run_regsho_script() -> str:
    proc = await asyncio.create_subprocess_exec(
        os.getenv("PYTHON", "python3"),
        os.path.join(os.path.dirname(__file__), "RegSHO_Tracker.py"),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_bytes, stderr_bytes = await proc.communicate()
    output = stdout_bytes.decode().strip()
    errors = stderr_bytes.decode().strip()
    if errors:
        output = f"{output}\n\nErrors:\n{errors}"
    return output

async def send_report(bot: discord.Bot, channel_id: int) -> None:
    channel = bot.get_channel(channel_id)
    if channel is None:
        raise RuntimeError(f"Could not find channel {channel_id}")
    report = await run_regsho_script()
    MAX_LENGTH = 2000
    for start in range(0, len(report), MAX_LENGTH):
        await channel.send(report[start:start + MAX_LENGTH])

def seconds_until_next_run(hour: int, minute: int, tz: Optional[ZoneInfo]) -> float:
    now = datetime.now(tz) if tz else datetime.now()
    target = datetime.combine(now.date(), dt_time(hour, minute), tzinfo=tz)
    if now >= target:
        target += timedelta(days=1)
    return (target - now).total_seconds()

async def scheduler(bot: discord.Bot, channel_id: int) -> None:
    tz = ZoneInfo("America/New_York") if ZoneInfo else None
    while True:
        await asyncio.sleep(seconds_until_next_run(3, 0, tz))
        try:
            await send_report(bot, channel_id)
        except Exception as exc:
            print(f"Error sending RegSHO report: {exc}")
        await asyncio.sleep(5)

class RegSHOClient(discord.Bot):
    """Discord bot that schedules the daily report and exposes a slash command."""

    def __init__(self, channel_id: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.channel_id = channel_id

    async def on_ready(self) -> None:
        print(f"Logged in as {self.user} (ID: {self.user.id})")
        self.loop.create_task(scheduler(self, self.channel_id))

    async def setup_hook(self) -> None:
        @self.command(
            name="regsho_tracker list",
            description="Run the RegSHO Threshold script and post the results.",
        )
        async def regsho_tracker_list(ctx: discord.ApplicationContext) -> None:
            await ctx.defer(ephemeral=True)
            try:
                await send_report(self, self.channel_id)
                await ctx.respond(
                    "RegSHO Threshold report has been posted to the configured channel."
                )
            except Exception as exc:
                await ctx.respond(f"Failed to post the report: {exc}", delete_after=30)
        # Register commands globally (takes up to an hour).  For instant testing,
        # pass guild IDs to sync() instead.
        await self.sync()

def main() -> None:
    token = os.environ.get("DISCORD_BOT_TOKEN")
    channel_id = os.environ.get("DISCORD_CHANNEL_ID")
    if not token:
        raise RuntimeError("DISCORD_BOT_TOKEN must be set.")
    if not channel_id:
        raise RuntimeError("DISCORD_CHANNEL_ID must be set.")
    intents = discord.Intents.default()
    bot = RegSHOClient(channel_id=int(channel_id), intents=intents)
    bot.run(token)

if __name__ == "__main__":
    main()