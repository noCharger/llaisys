import uuid
import time
from typing import List, Optional
from ..models.chat import ChatRequest, ChatResponse, Message
from ..dao.session_dao import SessionDAO
from ..services.scheduler import ClipperScheduler, RequestJob
from ..services.context_manager import ContextManager
import logging

logger = logging.getLogger("llaisys.chat_service")

class ChatService:
    def __init__(self, scheduler: ClipperScheduler, session_dao: SessionDAO, context_manager: ContextManager):
        self.scheduler = scheduler
        self.session_dao = session_dao
        self.context_manager = context_manager
        self.default_system_prompt = (
            "You are a helpful AI assistant.\n"
            "{% if context_items %}\n"
            "Context:\n"
            "{% for item in context_items %}\n"
            "- {{ item.content }}\n"
            "{% endfor %}\n"
            "Use the context above if relevant.\n"
            "{% endif %}\n"
            "Answer the user clearly. Adapt the length and detail of your response to the nature of the user's query: "
            "provide concise answers for simple questions, and detailed, step-by-step explanations for complex tasks."
        )

    async def chat_completion(self, request: ChatRequest, tenant_id: str) -> ChatResponse:
        session_id = request.session_id
        if not session_id:
            session = await self.session_dao.create_session(tenant_id)
            session_id = session.id
        else:
            session = await self.session_dao.get_session(session_id)
            if not session:
                session = await self.session_dao.create_session(tenant_id)
                session_id = session.id

        sys_prompt_content = request.system_prompt if request.system_prompt else self.default_system_prompt

        if request.use_template and sys_prompt_content:
            try:
                sys_prompt_content = self.context_manager.render_template(sys_prompt_content)
            except Exception as e:
                logger.error(f"Failed to render system prompt: {e}")

                pass

        user_msg = request.messages[-1]

        is_new_session = len(session.messages) == 0

        final_input_text = user_msg.content
        if is_new_session and sys_prompt_content:

             sys_msg = Message(role="system", content=sys_prompt_content)
             await self.session_dao.add_message(session_id, sys_msg)

             pass

        await self.session_dao.add_message(session_id, user_msg)

        input_text = user_msg.content
        if is_new_session and sys_prompt_content:
             input_text = f"System: {sys_prompt_content}\nUser: {user_msg.content}"

        input_ids = [ord(c) for c in input_text]

        req_id = str(uuid.uuid4())

        max_gen_tokens = request.max_tokens if request.max_tokens and request.max_tokens > 0 else 4096

        job = RequestJob(
            request_id=req_id,
            session_ptr=session_id,
            input_ids=input_ids,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_tokens=max_gen_tokens
        )

        await self.scheduler.submit(job)
        response_text = await job.future

        assistant_msg = Message(role="assistant", content=response_text)
        await self.session_dao.add_message(session_id, assistant_msg)

        return ChatResponse(
            id=req_id,
            created=int(time.time()),
            model=request.model,
            choices=[{
                "index": 0,
                "message": assistant_msg.model_dump(),
                "finish_reason": "stop"
            }],
            usage={"prompt_tokens": len(input_ids), "completion_tokens": len(response_text), "total_tokens": len(input_ids) + len(response_text)}
        )
