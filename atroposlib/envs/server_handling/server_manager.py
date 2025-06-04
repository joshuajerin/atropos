import asyncio
import inspect
import os
import socket
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Union

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.completion import Completion
from pydantic import BaseModel, Field

from atroposlib.envs.server_handling.openai_server import OpenAIServer
from atroposlib.envs.server_handling.server_baseline import (
    APIServer,
    APIServerConfig,
    ServerBaseline,
)
from atroposlib.envs.server_handling.server_harness import ServerHarness
from atroposlib.envs.server_handling.trl_vllm_server import TrlVllmServer


def is_port_available(port):
    """Check if a port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return True
        except socket.error:
            return False

def find_available_port(start_port=9000, max_attempts=100):
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        if is_port_available(port):
            return port
    raise RuntimeError(f"Could not find an available port after {max_attempts} attempts starting from {start_port}")

def get_port_range_start():
    """Get the starting port for server allocation from environment variable or use default."""
    return int(os.environ.get("ATROPOS_PORT_RANGE_START", 9000))

class ServerManagerConfig(BaseModel):
    slurm: bool = Field(
        default=False, description="Whether environment is running on slurm or not."
    )
    testing: bool = Field(
        default=False, description="If set to True, environment uses mock OpenAI data."
    )
    port_range_start: int = Field(
        default_factory=get_port_range_start, 
        description="Starting port for server allocation. Default is 9000 or ATROPOS_PORT_RANGE_START env var."
    )


class ServerManager:
    def __init__(
        self,
        configs: Union[ServerBaseline, List[APIServerConfig]],
        server_class: APIServer = APIServer,
        slurm=False,
        testing=False,
        port_range_start=None,
    ):
        # First we check to see if it's the base server class, and if so, we need to select the appropriate server class
        # You can't use type() to check if it's the base server class, because it's an abstract class, it'll appear as
        # an ABCMeta, not what you're expecting.
        if inspect.isabstract(server_class):
            if not isinstance(configs, list):
                if configs.server_type == "openai":
                    server_class = OpenAIServer
                elif configs.server_type == "trl":
                    server_class = TrlVllmServer
                else:
                    raise ValueError(f"Invalid server type: {configs.server_type}")
            else:
                if configs[0].server_type == "openai":
                    server_class = OpenAIServer
                elif configs[0].server_type == "trl":
                    server_class = TrlVllmServer
                else:
                    raise ValueError(f"Invalid server type: {configs[0].server_type}")
        # Set the port range start
        if port_range_start is None:
            self.port_range_start = get_port_range_start()
        else:
            self.port_range_start = port_range_start
            
        if testing:
            # testing :)
            self.servers = [ServerHarness()]
            return
        if not isinstance(configs, list):
            urls = []
            if os.environ.get("SLURM_JOB_NODELIST", None) is not None:
                nodelist = (
                    os.popen(
                        f'scontrol show hostnames {os.environ["SLURM_JOB_NODELIST"]}'
                    )
                    .read()
                    .split("\n")
                )
                nodelist = [node for node in nodelist if node != ""]
                if len(nodelist) < 2:
                    # localhost!
                    for i in range(4):
                        # Find available ports dynamically
                        port = find_available_port(self.port_range_start + i + 4)
                        urls.append(f"http://localhost:{port}/v1")
                else:
                    num_training_nodes = int(os.environ.get("NUM_TRAINING_NODES"))
                    for node in nodelist[num_training_nodes:]:
                        for i in range(8 // os.environ.get("INFER_TP", 1)):
                            # Use the configured port range start
                            urls.append(f"http://{node}:{self.port_range_start + i}/v1")
                openai_configs = []
            else:
                # localhost!
                for i in range(4):
                    # Find available ports dynamically for localhost
                    port = find_available_port(self.port_range_start + i + 4)
                    urls.append(f"http://localhost:{port}/v1")
                openai_configs = []
            for url in urls:
                openai_configs.append(
                    APIServerConfig(
                        base_url=url,
                        timeout=configs.timeout,
                        num_max_requests_at_once=configs.num_max_requests_at_once,
                        num_requests_for_eval=configs.num_requests_for_eval,
                        model_name=configs.model_name,
                        rolling_buffer_length=configs.rolling_buffer_length,
                        api_key="x",
                    )
                )
            self.servers = [server_class(config) for config in openai_configs]
        elif not slurm:
            self.servers = [server_class(config) for config in configs]
        else:
            nodelist = (
                os.popen(f'scontrol show hostnames {os.environ["SLURM_JOB_NODELIST"]}')
                .read()
                .split("\n")
            )
            nodelist = [node for node in nodelist if node != ""]
            if len(nodelist) < 2:
                print(
                    "Not enough nodes to distribute to, assuming single node"
                    " and you've setup your sglang appropriately."
                )
                self.servers = [server_class(config) for config in configs]
                return
            urls = []
            num_training_nodes = int(os.environ.get("NUM_TRAINING_NODES"))
            for node in nodelist[num_training_nodes:]:
                if node == "":
                    continue
                for i in range(8 // os.environ.get("INFER_TP", 1)):
                    # Use the configured port range start
                    urls.append(f"http://{node}:{self.port_range_start + i}/v1")
            # assume at least one good config is passed in
            new_configs = []
            for i in range(len(urls)):
                new_conf = configs[0].model_copy(deep=True)
                new_conf.base_url = urls[i]
                new_configs.append(new_conf)
            self.servers = [server_class(config) for config in new_configs]

    async def update_weight(self, weight: float):
        for server in self.servers:
            await server.update_weight(weight)

    async def wait_for_sem(self, is_training):
        """
        Wait for a server to be available. This is used to prevent the client from
        overwhelming the server with requests.
        """
        if is_training:
            eval_vals = [
                (
                    max(0, server.eval_sem._value - server.eval_sem.min_val())
                    if server.eval_sem._value != server.eval_sem.max_val
                    else 0
                )
                for server in self.servers
            ]
            sem_vals = [
                max(0, (server.sem._value - server.sem.min_val()) - eval_val)
                for server, eval_val in zip(self.servers, eval_vals)
            ]
        else:
            sem_vals = [
                max(0, server.eval_sem._value - server.eval_sem.min_val())
                for server in self.servers
            ]
        while all([sem_val <= 0 for sem_val in sem_vals]):
            # None available... wait
            await asyncio.sleep(1)

    async def chat_completion(self, **kwargs) -> ChatCompletion:
        is_train = kwargs.get("split", "train") == "train"
        most_available_server = 0
        most_available_server_num_slots = -1
        await self.wait_for_sem(is_train)
        for i, server in enumerate(self.servers):
            if not server.server_healthy:
                continue
            if (
                server.sem._value if is_train else server.eval_sem._value
            ) > most_available_server_num_slots:
                most_available_server = i
                most_available_server_num_slots = (
                    server.sem._value if is_train else server.eval_sem._value
                )
        return await self.servers[most_available_server].chat_completion(**kwargs)

    async def completion(self, **kwargs) -> Completion:
        is_train = kwargs.get("split", "train") == "train"
        most_available_server = 0
        most_available_server_num_slots = -1
        await self.wait_for_sem(is_train)
        for i, server in enumerate(self.servers):
            if not server.server_healthy:
                continue
            if (
                server.sem._value if is_train else server.eval_sem._value
            ) > most_available_server_num_slots:
                most_available_server = i
                most_available_server_num_slots = (
                    server.sem._value if is_train else server.eval_sem._value
                )
        return await self.servers[most_available_server].completion(**kwargs)

    @asynccontextmanager
    async def dedicated_server(self) -> AsyncGenerator[OpenAIServer, None]:
        most_available_server = 0
        most_available_server_num_slots = -1
        for i, server in enumerate(self.servers):
            if not server.server_healthy:
                continue
            if server.sem._value > most_available_server_num_slots:
                most_available_server = i
                most_available_server_num_slots = server.sem._value
        async with self.servers[most_available_server].sem:
            try:
                yield self.servers[most_available_server]
            finally:
                pass
