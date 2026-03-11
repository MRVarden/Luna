"""Orchestrator — cognitive loop and supporting infrastructure.

The CognitiveLoop is the persistent daemon that owns all cognitive
subsystems. ChatSession attaches/detaches from it.

Legacy modules (orchestrator.py, agent_registry.py) are deprecated and
retained only for test compatibility.
"""

from luna.orchestrator.cognitive_loop import CognitiveLoop, SessionHandle
from luna.orchestrator.message_bus import Message, MessageBus
from luna.orchestrator.retry import RetryPolicy, retry_async
from luna.orchestrator.task_queue import PrioritizedTask, TaskQueue

__all__ = [
    "CognitiveLoop",
    "Message",
    "MessageBus",
    "PrioritizedTask",
    "RetryPolicy",
    "SessionHandle",
    "TaskQueue",
    "retry_async",
]
