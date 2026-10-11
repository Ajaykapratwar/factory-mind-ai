"""
FactoryMind AI (Pharma) - In-Process Event Bus (§11.5)
Asynchronous event-driven architecture with idempotent event dispatching.
"""
from __future__ import annotations
import asyncio
from typing import Dict, Any, List, Callable, Coroutine, Set
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime


class Event(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(..., description="Unique event identifier")
    name: str = Field(..., description="Event type name")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    payload: Dict[str, Any] = Field(default_factory=dict)


class EventBus:
    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Event], Coroutine[Any, Any, None]]]] = {}
        self._processed_event_ids: Set[str] = set()
        self._event_history: List[Event] = []

    def subscribe(self, event_name: str, handler: Callable[[Event], Coroutine[Any, Any, None]]):
        if event_name not in self._subscribers:
            self._subscribers[event_name] = []
        self._subscribers[event_name].append(handler)

    async def publish(self, event: Event) -> int:
        """
        Publishes event to all registered subscribers.
        Idempotent: events with previously processed IDs are skipped.
        """
        if event.id in self._processed_event_ids:
            return 0  # Idempotent deduplication

        self._processed_event_ids.add(event.id)
        self._event_history.append(event)

        handlers = self._subscribers.get(event.name, [])
        handlers_all = self._subscribers.get("*", [])

        all_handlers = handlers + handlers_all
        if not all_handlers:
            return 0

        # Execute all async handlers
        await asyncio.gather(*[h(event) for h in all_handlers], return_exceptions=True)
        return len(all_handlers)

    def publish_sync(self, event: Event):
        """Synchronous helper for background event publication."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.publish(event))
            else:
                loop.run_until_complete(self.publish(event))
        except RuntimeError:
            asyncio.run(self.publish(event))

    def get_history(self, limit: int = 100) -> List[Event]:
        return self._event_history[-limit:]
