"""

BOT_NAME="AlgoSearch"; modal deploy --name $BOT_NAME bot_${BOT_NAME}.py; curl -X POST https://api.poe.com/bot/fetch_settings/$BOT_NAME/$POE_ACCESS_KEY

"""
from __future__ import annotations

from typing import AsyncIterable
import os

import fastapi_poe as fp
from modal import Image, Stub, asgi_app
from fastapi_poe.client import MetaMessage, ProtocolMessage

from src.embedder import VectorDB, get_embeddings
from src.utils import read_problems, problems_filenames


USER_PROMPT = """
I have the following competitive programming problem that I want to show someone else:
[[ORIGINAL]]
Can you strip off all the stories, legends, characters, backgrounds etc. from the statement while still enabling everyone to understand the problem. This is to say, do not remove anything necessary to understand the full problem and one should feel safe to replace the original statement with your version of the statement. If it is not in English make it English. Provide the simplified statement directly without jargon. Use mathjax ($...$) for math.
""".strip()


topk = 5
db = VectorDB().load()
emb_keys = set([x[0] for x in db.metadata])
print("read", len(emb_keys), "embeddings from db")
problems = {}
for f in problems_filenames():
    for p in read_problems("problems/" + f):
        problems[p["uid"]] = p
print("read", len(problems), "problems from db")


class GPT35TurboAllCapsBot(fp.PoeBot):
    async def get_response(
        self, request: fp.QueryRequest
    ) -> AsyncIterable[fp.PartialResponse]:
        user_message = request.query[-1].content
        wrapped_message = USER_PROMPT.replace("[[ORIGINAL]]", user_message)

        request.query = [
            ProtocolMessage(role="system", content=wrapped_message),
        ]

        current_bot_reply = "Your summary has been paraphrased as follows:\n\n"
        yield self.text_event(current_bot_reply)

        async for msg in fp.stream_request(
            request, "GPT-3.5-Turbo", request.access_key
        ):
            if isinstance(msg, MetaMessage):
                continue
            elif msg.is_suggested_reply:
                yield self.suggested_reply_event(msg.text)
            elif msg.is_replace_response:
                yield self.replace_response_event(msg.text)
            else:
                current_bot_reply += msg.text
                yield self.replace_response_event(current_bot_reply.replace("$", "$$"))

        emb = get_embeddings([current_bot_reply])[0]
        # query nearest
        nearest = db.query_nearest(emb, k=topk)

        for similarity, metadata in nearest:
            summary, uid = metadata
            print(similarity)
            print(metadata)
            statement = problems[uid]["statement"].replace("\n", "\n\n")
            summary = sorted(problems[uid]["processed"], key=lambda t: t["template_md5"])
            summary = summary[0]["result"].replace("$", "$$")
            statement = statement.replace("$", "$$")
            print(summary)
            print(statement)

            title = uid  # problems[uid]['title']
            url = problems[uid]["url"]
            markdown = f"\n\n## [{title}]({url}) ({similarity:.2f})\n\n"
            if summary is not None:
                markdown += f"{summary}\n\n"
            # markdown += f"### Statement\n\n{statement}"
            yield self.text_event(markdown)

    async def get_settings(self, setting: fp.SettingsRequest) -> fp.SettingsResponse:
        return fp.SettingsResponse(
            server_bot_dependencies={"GPT-3.5-Turbo": 1},
            introduction_message="Please provide a summary of a competitive programming problem."
        )


REQUIREMENTS = [
    "fastapi-poe==0.0.24",
    "beautifulsoup4==4.11.1",
    "cohere==4.34",
    "GitPython==3.1.40",
    "gradio==4.3.0",
    "numpy==1.23.1",
    "openai==1.3.2",
    "Requests==2.31.0",
    "tqdm==4.64.1",
]
image = Image.debian_slim().pip_install(*REQUIREMENTS).env(
    {
        "COHERE_API_KEY": os.environ["COHERE_API_KEY"],
        "POE_ACCESS_KEY": os.environ["POE_ACCESS_KEY"],
    }
).copy_local_dir(
    "problems/", "/root/problems/"
).copy_local_dir(
    "embs/", "/root/embs/"
).copy_local_file(
    "settings.json", "/root/settings.json"
)
stub = Stub("turbo-allcaps-poe")


@stub.function(image=image)
@asgi_app()
def fastapi_app():
    bot = GPT35TurboAllCapsBot()
    # Optionally, provide your Poe access key here:
    # 1. You can go to https://poe.com/create_bot?server=1 to generate an access key.
    # 2. We strongly recommend using a key for a production bot to prevent abuse,
    # but the starter examples disable the key check for convenience.
    # 3. You can also store your access key on modal.com and retrieve it in this function
    # by following the instructions at: https://modal.com/docs/guide/secrets
    # POE_ACCESS_KEY = ""
    # app = make_app(bot, access_key=POE_ACCESS_KEY)
    app = fp.make_app(bot, allow_without_key=True)
    return app
