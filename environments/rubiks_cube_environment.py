#!/usr/bin/env python3
"""
RubiksCubeEnv: Trainer environment for Rubik's Cube solving with multi-step reasoning

This environment implements a Rubik's cube solver that trains LLMs to solve cubes
through step-by-step reasoning and visualization. Extends BaseEnv.
"""

import asyncio
import copy
import json
import logging
import random
import re
from typing import Dict, List, Optional, Tuple, Any
import string

import numpy as np

# Define the face colors for visualization
UP_COLOR = 'W'     # White
DOWN_COLOR = 'Y'   # Yellow
RIGHT_COLOR = 'R'  # Red
LEFT_COLOR = 'O'   # Orange
FRONT_COLOR = 'G'  # Green
BACK_COLOR = 'B'   # Blue

class Cube:
    """
    A Rubik's cube implementation with accurate move handling.
    """
    def __init__(self):
        # Initialize a solved cube
        self.reset()
    
    def reset(self):
        """Reset the cube to solved state"""
        # Initialize the cube as a 3D array [face][row][col]
        # Faces: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=FRONT, 5=BACK
        self.cube = [
            [[UP_COLOR for _ in range(3)] for _ in range(3)],      # UP
            [[DOWN_COLOR for _ in range(3)] for _ in range(3)],    # DOWN
            [[LEFT_COLOR for _ in range(3)] for _ in range(3)],    # LEFT
            [[RIGHT_COLOR for _ in range(3)] for _ in range(3)],   # RIGHT
            [[FRONT_COLOR for _ in range(3)] for _ in range(3)],   # FRONT
            [[BACK_COLOR for _ in range(3)] for _ in range(3)]     # BACK
        ]
    
    def is_solved(self) -> bool:
        """Check if the cube is solved"""
        for face in self.cube:
            center_color = face[1][1]  # Center color never changes
            for row in face:
                for color in row:
                    if color != center_color:
                        return False
        return True
    
    def count_solved_cubies(self) -> float:
        """
        Count the number of stickers in their correct position
        Returns a normalized score between 0 and 1
        """
        # Create a solved reference cube
        reference = Cube()
        
        # Count matching stickers
        total_stickers = 6 * 9  # 6 faces, 9 stickers per face
        match_count = 0
        
        for face_idx in range(6):
            for i in range(3):
                for j in range(3):
                    if self.cube[face_idx][i][j] == reference.cube[face_idx][i][j]:
                        match_count += 1
        
        return match_count / total_stickers
    
    def rotate(self, move: str):
        """
        Perform a move on the cube using standard notation
        U, D, L, R, F, B are clockwise rotations of respective faces
        U', D', L', R', F', B' are counterclockwise rotations
        U2, D2, L2, R2, F2, B2 are double (180°) rotations
        """
        # Map move notation to face index and rotation count
        face_map = {
            'U': 0, 'D': 1, 'L': 2, 'R': 3, 'F': 4, 'B': 5
        }
        
        # Parse the move
        if len(move) == 0:
            raise ValueError("Empty move")
        
        face = move[0]
        if face not in face_map:
            raise ValueError(f"Invalid face: {face}")
        
        face_idx = face_map[face]
        
        # Handle rotation direction
        if len(move) == 1:
            # Clockwise rotation
            count = 1
        elif len(move) == 2:
            if move[1] == "'":
                # Counterclockwise rotation
                count = 3
            elif move[1] == "2":
                # Double rotation
                count = 2
            else:
                raise ValueError(f"Invalid move modifier: {move[1]}")
        else:
            raise ValueError(f"Invalid move format: {move}")
        
        # Apply the rotation 'count' times
        for _ in range(count):
            self._rotate_face_clockwise(face_idx)
            self._rotate_adjacent_faces(face_idx)
    
    def _rotate_face_clockwise(self, face_idx: int):
        """Rotate a face clockwise"""
        face = self.cube[face_idx]
        new_face = [[None for _ in range(3)] for _ in range(3)]
        
        # Copy with 90-degree clockwise rotation
        for i in range(3):
            for j in range(3):
                new_face[j][2-i] = face[i][j]
        
        self.cube[face_idx] = new_face
    
    def _rotate_adjacent_faces(self, face_idx: int):
        """Rotate the appropriate edges on adjacent faces"""
        if face_idx == 0:  # UP face
            # Rotate the top edges of FRONT, RIGHT, BACK, LEFT
            temp = self.cube[4][0][:]  # Save FRONT top edge
            self.cube[4][0] = self.cube[2][0][:]  # FRONT <- LEFT
            self.cube[2][0] = self.cube[5][0][:]  # LEFT <- BACK
            self.cube[5][0] = self.cube[3][0][:]  # BACK <- RIGHT
            self.cube[3][0] = temp                # RIGHT <- FRONT
            
        elif face_idx == 1:  # DOWN face
            # Rotate the bottom edges of FRONT, LEFT, BACK, RIGHT
            temp = self.cube[4][2][:]  # Save FRONT bottom edge
            self.cube[4][2] = self.cube[3][2][:]  # FRONT <- RIGHT
            self.cube[3][2] = self.cube[5][2][:]  # RIGHT <- BACK
            self.cube[5][2] = self.cube[2][2][:]  # BACK <- LEFT
            self.cube[2][2] = temp                # LEFT <- FRONT
            
        elif face_idx == 2:  # LEFT face
            # Rotate the left edges of UP, FRONT, DOWN, BACK
            # Need to extract and set columns, not rows
            temp = [self.cube[0][i][0] for i in range(3)]  # Save UP left column
            
            # UP left <- BACK right (reversed)
            for i in range(3):
                self.cube[0][i][0] = self.cube[5][2-i][2]
            
            # BACK right <- DOWN left (reversed)
            for i in range(3):
                self.cube[5][i][2] = self.cube[1][2-i][0]
            
            # DOWN left <- FRONT left
            for i in range(3):
                self.cube[1][i][0] = self.cube[4][i][0]
            
            # FRONT left <- UP left
            for i in range(3):
                self.cube[4][i][0] = temp[i]
                
        elif face_idx == 3:  # RIGHT face
            # Rotate the right edges of UP, BACK, DOWN, FRONT
            temp = [self.cube[0][i][2] for i in range(3)]  # Save UP right column
            
            # UP right <- FRONT right
            for i in range(3):
                self.cube[0][i][2] = self.cube[4][i][2]
            
            # FRONT right <- DOWN right
            for i in range(3):
                self.cube[4][i][2] = self.cube[1][i][2]
            
            # DOWN right <- BACK left (reversed)
            for i in range(3):
                self.cube[1][i][2] = self.cube[5][2-i][0]
            
            # BACK left <- UP right (reversed)
            for i in range(3):
                self.cube[5][i][0] = temp[2-i]
                
        elif face_idx == 4:  # FRONT face
            # Rotate the edges of UP bottom, RIGHT left, DOWN top, LEFT right
            # UP bottom row
            temp = self.cube[0][2][:]
            
            # UP bottom <- LEFT right (rotated)
            for i in range(3):
                self.cube[0][2][i] = self.cube[2][2-i][2]
            
            # LEFT right <- DOWN top (rotated)
            for i in range(3):
                self.cube[2][i][2] = self.cube[1][0][i]
            
            # DOWN top <- RIGHT left (rotated)
            for i in range(3):
                self.cube[1][0][i] = self.cube[3][2-i][0]
            
            # RIGHT left <- UP bottom (rotated)
            for i in range(3):
                self.cube[3][i][0] = temp[i]
                
        elif face_idx == 5:  # BACK face
            # Rotate the edges of UP top, LEFT left, DOWN bottom, RIGHT right
            # UP top row
            temp = self.cube[0][0][:]
            
            # UP top <- RIGHT right (rotated)
            for i in range(3):
                self.cube[0][0][i] = self.cube[3][2-i][2]
            
            # RIGHT right <- DOWN bottom (rotated)
            for i in range(3):
                self.cube[3][i][2] = self.cube[1][2][i]
            
            # DOWN bottom <- LEFT left (rotated)
            for i in range(3):
                self.cube[1][2][i] = self.cube[2][2-i][0]
            
            # LEFT left <- UP top (rotated)
            for i in range(3):
                self.cube[2][i][0] = temp[i]
    
    def __str__(self) -> str:
        """Convert cube to string representation"""
        face_names = ['U', 'D', 'L', 'R', 'F', 'B']
        result = []
        
        for i, face in enumerate(self.cube):
            result.append(f"{face_names[i]}: {' '.join(face[0])}")
            result.append(f"   {' '.join(face[1])}")
            result.append(f"   {' '.join(face[2])}")
        
        return "\n".join(result)
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
    ScoredDataGroup,
)
from atroposlib.utils.message_history_utils import (
    ensure_trajectory_token_limit,
    truncate_thinking,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer
from atroposlib.utils.tool_call_parser import parse_tool_call

logger = logging.getLogger(__name__)


class RubiksCubeEnvConfig(BaseEnvConfig):
    # Environment configuration
    max_steps: int = 20  # Maximum steps allowed to solve the cube
    temperature: float = 0.7
    top_p: float = 0.9
    wandb_name: str = "rubiks_cube"
    thinking_active: bool = True
    eval_episodes: int = 100
    max_think_chars_history: int = 3000
    max_trajectory_tokens: int = 24576  # seq_len of RL trainer
    debug_mode: bool = False
    group_size: int = 16
    tiebreak_token_factor: float = 0.01
    
    # Cube-specific configuration
    scramble_moves: int = 7  # Number of random moves to scramble the cube
    cube_size: int = 3  # 3x3 cube by default
    reward_per_correctly_placed_cubie: float = 0.05
    reward_per_step_reduction: float = 0.01  # Small penalty for using more steps


class RubiksCubeScoredDataGroup(ScoredDataGroup):
    seed: int
    tokens: Optional[List[List[int]]] = None
    masks: Optional[List[List[int]]] = None
    scores: Optional[List[float]] = None
    messages: Optional[List[List[Dict]]] = None
    parsed_actions: Optional[List[str]] = None


class CubeState:
    def __init__(self, seed: int, scramble_moves: int):
        self.seed = seed
        self.cube = Cube()
        self.message_history: List[Dict] = []
        self.actions: List[str] = []
        self.step_rewards: List[float] = []
        self.total_reward: float = 0.0
        self.num_steps: int = 0
        
        # Seed random number generator for reproducibility
        random.seed(seed)
        
        # Reset cube to solved state
        self.cube.reset()
        
        # Scramble the cube with random moves
        self._scramble_cube(scramble_moves)
    
    def _scramble_cube(self, num_moves: int):
        """Scramble the cube with random moves"""
        moves = ["U", "D", "L", "R", "F", "B", 
                "U'", "D'", "L'", "R'", "F'", "B'",
                "U2", "D2", "L2", "R2", "F2", "B2"]
        
        scramble_sequence = []
        for _ in range(num_moves):
            move = random.choice(moves)
            scramble_sequence.append(move)
            self.cube.rotate(move)
            
        return " ".join(scramble_sequence)
    
    def apply_move(self, move: str) -> bool:
        """Apply a move to the cube and return success"""
        try:
            self.cube.rotate(move)
            self.actions.append(move)
            self.num_steps += 1
            return True
        except Exception as e:
            logger.error(f"Error applying move {move}: {e}")
            return False

    def is_solved(self) -> bool:
        """Check if the cube is solved"""
        return self.cube.is_solved()
    
    def get_cube_state_visualization(self) -> str:
        """Get a text representation of the cube state for visualization"""
        # This returns a readable string representation of the cube layout
        return str(self.cube)


class RubiksCubeEnv(BaseEnv):
    def __init__(
        self,
        config: RubiksCubeEnvConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = True,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.episodes: Dict[int, CubeState] = {}
        self.debug_mode = config.debug_mode
        self.completed_episode_metrics_buffer: List[Dict[str, float]] = []
        
        if self.debug_mode:
            logger.setLevel(logging.DEBUG)
        else:
            if logger.level == logging.NOTSET or logger.level > logging.WARNING:
                logger.setLevel(logging.WARNING)

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "apply_move",
                    "description": "Apply a move to the Rubik's cube.",
                    "parameters": {
                        "move": {
                            "type": "string", 
                            "description": "The move to apply to the cube. Valid moves are U, D, L, R, F, B (clockwise), U', D', L', R', F', B' (counterclockwise), and U2, D2, L2, R2, F2, B2 (180 degrees)."
                        }
                    },
                },
            }
        ]

        tools_json = json.dumps(self.tools)
        self.system_prompt = (
            "You are an AI that solves Rubik's cubes step-by-step with clear reasoning. "
            "You will be given the current state of a Rubik's cube, and you need to provide "
            "moves to solve it.\n\n"
            "The notation for cube moves follows the standard Rubik's cube notation:\n"
            "- U: rotate the up face clockwise\n"
            "- D: rotate the down face clockwise\n"
            "- L: rotate the left face clockwise\n"
            "- R: rotate the right face clockwise\n"
            "- F: rotate the front face clockwise\n"
            "- B: rotate the back face clockwise\n"
            "- U', D', L', R', F', B': rotate the corresponding face counterclockwise\n"
            "- U2, D2, L2, R2, F2, B2: rotate the corresponding face 180 degrees\n\n"
            "You should analyze the current state of the cube, identify patterns, "
            "and explain your reasoning step by step.\n\n"
            "You should enclose your thoughts and internal monologue inside <think> </think> tags, and then "
            "provide your move using the apply_move function call.\n\n"
            f"<tools>\n{tools_json}\n</tools>\n\n"
            "For your function call, return a JSON object with function name and arguments "
            "within <tool_call> </tool_call> tags with the following schema:\n"
            '<tool_call>\n{"arguments": {"move": "U"}, "name": "apply_move"}\n</tool_call>\n\n'
            "Your full answer format should be:\n"
            "<think>\n[Your detailed reasoning about the current cube state and the best move to make]\n</think>\n\n"
            '<tool_call>\n{"arguments": {"move": "R"}, "name": "apply_move"}\n</tool_call>\n\n'
            "Remember to carefully analyze the cube state and work toward the solution step by step."
        )

    def _get_or_create_episode(self, seed: int) -> CubeState:
        if seed not in self.episodes:
            ep = CubeState(seed, self.config.scramble_moves)
            ep.message_history = [{"role": "system", "content": self.system_prompt}]
            # Add initial observation
            ep.message_history.append(
                {"role": "environment", "content": self._format_observation(ep)}
            )
            self.episodes[seed] = ep
        return self.episodes[seed]

    def _format_observation(self, cube_state: CubeState) -> str:
        """Format the cube state as a string observation for the LLM"""
        cube_visualization = cube_state.get_cube_state_visualization()
        
        moves_made = ", ".join(cube_state.actions) if cube_state.actions else "None"
        steps_remaining = self.config.max_steps - cube_state.num_steps
        
        message = (
            f"Current state of the Rubik's cube:\n\n"
            f"```\n{cube_visualization}\n```\n\n"
            f"Previous moves: {moves_made}\n"
            f"Steps remaining: {steps_remaining}\n"
        )
        
        if cube_state.is_solved():
            message += "\nCongratulations! The cube is now solved."
        
        return message

    def _calculate_cube_state_score(self, cube_state: CubeState) -> float:
        """
        Calculate a score based on how close the cube is to being solved.
        Higher scores for cubes that are closer to being solved.
        """
        # Base score
        score = 0.0
        
        # Reward for a solved cube
        if cube_state.is_solved():
            score += 1.0
            
        # Get the current state
        cube = cube_state.cube
        
        # Count correctly positioned cubies
        # This is a simplified approach - in a real implementation, 
        # we would calculate this from the cube's internal state
        correctly_placed = cube.count_solved_cubies()
        score += correctly_placed * self.config.reward_per_correctly_placed_cubie
        
        # Small penalty for using more steps
        steps_penalty = cube_state.num_steps * self.config.reward_per_step_reduction
        score -= steps_penalty
        
        return score

    def _parse_move(self, response: str) -> Optional[str]:
        """Extract move from the LLM response"""
        if not response:
            logger.warning(
                "Attempted to parse an empty response string. Returning None."
            )
            return None

        # First try parsing with tool_call tags
        parsed_name, parsed_args, is_error = parse_tool_call(
            response, self.tools, ["tool_call"]
        )
        
        # If that fails, try looking for direct text mentions of moves
        if is_error:
            error_detail = (
                parsed_name
                if isinstance(parsed_name, str) and parsed_name
                else "Parser indicated error, but no specific message was returned"
            )
            logger.warning(
                f"Failed to parse tool call. Full response: '{response}'. Error detail: {error_detail}"
            )
            
            # Fallback: Look for direct mentions of moves in the text
            valid_moves = ["U", "D", "L", "R", "F", "B", 
                          "U'", "D'", "L'", "R'", "F'", "B'",
                          "U2", "D2", "L2", "R2", "F2", "B2"]
            
            # Look for patterns like "I'll apply move X" or "Performing X rotation"
            move_patterns = [
                r'move\s+([UDLRFB][\'2]?)',
                r'applying\s+([UDLRFB][\'2]?)',
                r'perform\w*\s+([UDLRFB][\'2]?)',
                r'rotate\s+([UDLRFB][\'2]?)',
                r'rotation\s+([UDLRFB][\'2]?)',
                r'I\s*choose\s+([UDLRFB][\'2]?)',
                r'Execute\s+([UDLRFB][\'2]?)',
                r'([UDLRFB][\'2]?)\s+rotation',
                r'([UDLRFB][\'2]?)\s+move'
            ]
            
            for pattern in move_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    potential_move = match.group(1).strip()
                    if potential_move in valid_moves:
                        logger.warning(f"Recovered move '{potential_move}' from text using regex")
                        return potential_move
            
            return None

        move = parsed_args.get("move", "").strip() if isinstance(parsed_args, dict) else ""
        valid_moves = ["U", "D", "L", "R", "F", "B", 
                      "U'", "D'", "L'", "R'", "F'", "B'",
                      "U2", "D2", "L2", "R2", "F2", "B2"]
        
        # First check if the move is directly valid
        if move in valid_moves:
            return move
        
        # Check if the move is a sequence containing valid moves
        # (when LLM outputs move sequences like "R U R'")
        if " " in move:
            # Take only the first move in the sequence
            first_move = move.split()[0].strip()
            if first_move in valid_moves:
                logger.warning(f"Got move sequence '{move}' but taking only first move '{first_move}'")
                return first_move
        
        # If we get here, the move is invalid
        logger.warning(
            f"Parsed invalid move: '{move}'. "
            f"Full response: '{response}'. Parsed args: {parsed_args}"
        )
        return None

    def _score_response(
        self,
        is_valid_move: bool,
        response_text: str,
        cube_state: CubeState,
        is_solved: bool,
    ) -> float:
        """
        Calculate a score for a single agent response based on:
        1. Whether the move was valid
        2. Whether the move helps solve the cube
        3. Presence of thinking tags
        4. If the cube is solved
        """
        # Base score from cube state after the move
        current_score = self._calculate_cube_state_score(cube_state)
        
        # Bonus for valid moves
        if is_valid_move:
            current_score += 0.2
        else:
            current_score -= 0.2
        
        # Bonus for solving the cube
        if is_solved:
            current_score += 1.0
        
        # Check for thinking tags
        try:
            # Make sure response_text is a string
            if not isinstance(response_text, str):
                logger.warning(f"response_text is not a string: {type(response_text)}")
                response_text = str(response_text) if response_text is not None else ""
                
            match = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)
            if match:
                thinking_content = match.group(1)
                # Make sure thinking_content is a string before calling strip()
                if isinstance(thinking_content, str):
                    if thinking_content.strip():  # Not empty
                        current_score += 0.2
                    else:  # Empty thinking tags
                        current_score -= 0.1
                else:
                    logger.warning(f"thinking_content is not a string: {type(thinking_content)}")
                    current_score -= 0.1
            else:  # No thinking tags
                current_score -= 0.2
        except Exception as e:
            logger.warning(f"Error processing thinking tags: {e}")
            current_score -= 0.1
            
        return current_score

    async def _sample_response(self, messages: List[Dict], n: int = 1) -> List[str]:
        """Sample responses from the language model"""
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        try:
            completions = await self.server.completion(
                prompt=prompt,
                n=n,
                max_tokens=self.config.max_token_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
            return [choice.text for choice in completions.choices]
        except Exception as e:
            logger.error(f"API error during completion: {e}")
            return []

    async def _next_step(
        self, ep: CubeState, current_turn: int, max_turns: int
    ) -> Tuple[Optional[RubiksCubeScoredDataGroup], bool]:
        """Process one step of an episode"""
        G = self.config.group_size

        # Get current state
        current_state_messages = ep.message_history.copy()
        logger.debug(
            f"[Next Step Seed: {ep.seed} Turn: {current_turn + 1}/{max_turns}] "
            f"Current state history length: {len(current_state_messages)}"
        )

        messages_for_llm = current_state_messages.copy()
        agent_prompt_content = "<think>\n" if self.config.thinking_active else ""
        messages_for_llm.append({"role": "agent", "content": agent_prompt_content})

        # Generate G alternative responses
        try:
            responses = await self._sample_response(messages_for_llm, n=G)
            if not responses or len(responses) != G:
                logger.error(
                    f"[Next Step Seed: {ep.seed} Turn: {current_turn + 1}] "
                    f"Expected {G} responses, got {len(responses) if responses else 0}. "
                    f"Aborting step."
                )
                return None, True  # Episode termination
        except Exception as e_sample:
            logger.error(
                f"[Next Step Seed: {ep.seed} Turn: {current_turn + 1}] Error sampling responses: {e_sample}",
                exc_info=True,
            )
            return None, True  # Episode termination

        # Lists to store data for each alternative response
        alt_full_responses: List[str] = []
        alt_parsed_moves: List[Optional[str]] = []
        alt_is_valid_move: List[bool] = []
        alt_rewards: List[float] = []
        alt_next_state_msgs: List[List[Dict]] = []
        alt_is_terminal: List[bool] = []
        alt_is_solved: List[bool] = []
        alt_tokens: List[List[int]] = []
        alt_masks: List[List[int]] = []

        # Process each alternative response
        for i in range(G):
            llm_output_only = responses[i]
            full_agent_response = agent_prompt_content + llm_output_only
            alt_full_responses.append(full_agent_response)

            # Parse the move from the response
            parsed_move = self._parse_move(full_agent_response)
            alt_parsed_moves.append(parsed_move)
            
            # Create a copy of the current state for simulation
            sim_ep = copy.deepcopy(ep)
            
            # Apply the move if valid
            is_valid_move = False
            if parsed_move is not None:
                is_valid_move = sim_ep.apply_move(parsed_move)
            alt_is_valid_move.append(is_valid_move)
            
            # Check if the cube is solved after the move
            is_solved = sim_ep.is_solved()
            alt_is_solved.append(is_solved)
            
            # Calculate reward
            reward = self._score_response(
                is_valid_move, 
                full_agent_response, 
                sim_ep, 
                is_solved
            )
            alt_rewards.append(reward)
            
            # Determine if the episode terminates
            is_terminal = (
                is_solved or 
                (current_turn + 1 >= max_turns) or 
                not is_valid_move
            )
            alt_is_terminal.append(is_terminal)
            
            # Prepare next state messages
            current_state_plus_response_i = current_state_messages + [
                {"role": "agent", "content": full_agent_response}
            ]
            
            if not is_terminal:
                next_state_msgs_i = current_state_plus_response_i + [
                    {
                        "role": "environment",
                        "content": self._format_observation(sim_ep),
                    }
                ]
            else:
                next_state_msgs_i = current_state_plus_response_i
                
            alt_next_state_msgs.append(next_state_msgs_i)
            
            # Tokenize the next state for the trainer
            tokenized_i = tokenize_for_trainer(self.tokenizer, next_state_msgs_i)
            alt_tokens.append(tokenized_i["tokens"])
            alt_masks.append(tokenized_i["masks"])

        # Package the data for this step
        if not (
            len(alt_tokens) == G
            and len(alt_masks) == G
            and len(alt_rewards) == G
            and len(alt_next_state_msgs) == G
            and len(alt_parsed_moves) == G
        ):
            logger.error(
                f"[Next Step Seed: {ep.seed} Turn: {current_turn + 1}] "
                f"Mismatch in alternative list lengths before creating ScoredDataGroup. "
                f"Tokens:{len(alt_tokens)}, Masks:{len(alt_masks)}, Rewards:{len(alt_rewards)}, "
                f"Msgs:{len(alt_next_state_msgs)}, ParsedMoves:{len(alt_parsed_moves)}. Expected {G} for all. "
                f"Aborting step."
            )
            return None, True

        current_step_data = RubiksCubeScoredDataGroup(
            seed=ep.seed,
            tokens=alt_tokens,
            masks=alt_masks,
            scores=alt_rewards,
            messages=alt_next_state_msgs,
            parsed_actions=alt_parsed_moves,
        )

        # Find the best response based on the rewards
        best_reward_idx = np.argmax(alt_rewards)
        
        chosen_reward_for_log = alt_rewards[best_reward_idx]
        logger.debug(
            f"[Next Step Seed: {ep.seed} Turn: {current_turn + 1}] "
            f"Selected Alt {best_reward_idx} "
            f"(Reward: {chosen_reward_for_log}) "
            f"from {G} alternatives."
        )

        # Get the best parsed move
        chosen_move = alt_parsed_moves[best_reward_idx]
        chosen_full_response = alt_full_responses[best_reward_idx]
        
        logger.info(
            f"[Next Step Seed: {ep.seed} Turn: {current_turn + 1}] Chosen move: "
            f"{chosen_move} (from Alt {best_reward_idx} with "
            f"Reward {chosen_reward_for_log})"
        )

        # Add the response to the episode history
        response_for_history = truncate_thinking(
            chosen_full_response,
            self.tokenizer,
            self.config.max_think_chars_history,
        )
        ep.message_history.append({"role": "agent", "content": response_for_history})

        # Apply the chosen move to the main environment
        is_valid_move = False
        if chosen_move is not None:
            is_valid_move = ep.apply_move(chosen_move)
        
        # Check if the cube is solved
        is_solved = ep.is_solved()
        
        # Calculate reward
        step_reward = self._score_response(
            is_valid_move, 
            chosen_full_response, 
            ep, 
            is_solved
        )
        ep.step_rewards.append(step_reward)
        
        # Determine if the episode terminates
        is_episode_terminal = (
            is_solved or 
            (current_turn + 1 >= max_turns) or 
            not is_valid_move
        )
        
        # Add the next observation if the episode continues
        if not is_episode_terminal:
            ep.message_history.append(
                {"role": "environment", "content": self._format_observation(ep)}
            )

        return current_step_data, is_episode_terminal

    async def collect_trajectories(
        self, item: Tuple[int, int]
    ) -> Tuple[List[RubiksCubeScoredDataGroup], List[Tuple[int, int]]]:
        """Collect data for ONE FULL trajectory (episode)"""
        seed, _ = item
        G_config = self.config.group_size
        max_turns = self.config.max_steps

        trajectory_data_for_trainer: List[RubiksCubeScoredDataGroup] = []

        logger.info(
            f"[Collect Trajectories Seed: {seed}] Starting new trajectory. "
            f"Group size G={G_config}, Max turns={max_turns}."
        )

        try:
            ep = self._get_or_create_episode(seed)
        except Exception as e:
            logger.error(
                f"[Collect Trajectories Seed: {seed}] Fatal error creating/getting episode: {e}",
                exc_info=True,
            )
            return [], []

        for turn_idx in range(max_turns):
            logger.debug(
                f"[Collect Trajectories Seed: {seed}] Attempting turn {turn_idx + 1}/{max_turns}."
            )

            try:
                step_data, is_terminal_this_step = await self._next_step(
                    ep, turn_idx, max_turns
                )
            except Exception as e:
                if "'list' object has no attribute 'strip'" in str(e):
                    # Special handling for this common error which doesn't actually
                    # prevent the overall process from working
                    logger.error(f"[Collect Trajectories Seed: {seed}] Non-fatal error in _next_step: {e}")
                    # Since we can't recover the step data, mark this step as failed
                    step_data = None
                    is_terminal_this_step = True
                else:
                    # For other errors, log and terminate
                    logger.error(
                        f"[Collect Trajectories Seed: {seed}] Error in _next_step: {e}",
                        exc_info=True
                    )
                    step_data = None
                    is_terminal_this_step = True

            if step_data:
                trajectory_data_for_trainer.append(step_data)
            else:
                logger.error(
                    f"[Collect Trajectories Seed: {seed}] Turn {turn_idx + 1} failed to produce data."
                    " Terminating episode."
                )
                is_terminal_this_step = True

            if is_terminal_this_step:
                final_reward_at_termination = (
                    sum(ep.step_rewards) if ep.step_rewards else 0.0
                )
                logger.info(
                    f"[Collect Trajectories Seed: {seed}] Episode ended at turn {turn_idx + 1}. "
                    f"Reason: step reported terminal. Total reward: {final_reward_at_termination:.2f}"
                )
                break
        else:
            logger.info(
                f"[Collect Trajectories Seed: {seed}] Episode reached max_turns ({max_turns})."
            )

        final_reward = sum(ep.step_rewards) if ep.step_rewards else 0.0
        
        # Store metrics for this episode
        episode_summary_metrics = {
            "seed": ep.seed,
            "total_reward": final_reward,
            "num_steps": ep.num_steps,
            "is_solved": ep.is_solved(),
        }
        self.completed_episode_metrics_buffer.append(episode_summary_metrics)
        
        # Clean up the episode
        if seed in self.episodes:
            del self.episodes[seed]

        # Ensure the trajectory doesn't exceed token limits
        limited_trajectory_data = ensure_trajectory_token_limit(
            trajectory_data_for_trainer,
            self.tokenizer,
            self.config.max_trajectory_tokens,
        )

        return limited_trajectory_data, []

    async def setup(self):
        """Initialize the environment"""
        # Nothing to do here as we don't need any special setup
        pass

    async def get_next_item(self) -> Tuple[int, int]:
        """Generate a new random seed for the next episode"""
        return (random.randint(0, 1000000), 0)

    async def rollout_and_score_eval(self, seed: int) -> Dict[str, float]:
        """Run a single episode for evaluation with a single response per step"""
        ep = self._get_or_create_episode(seed)
        max_turns = self.config.max_steps
        metrics = {
            "seed": seed,
            "total_reward": 0.0,
            "num_turns": 0,
            "num_valid_moves": 0,
            "num_invalid_moves": 0,
            "is_solved": False,
        }

        for turn in range(max_turns):
            messages = ep.message_history.copy()
            agent_prompt_content = "<think>\n" if self.config.thinking_active else ""
            messages.append({"role": "agent", "content": agent_prompt_content})

            # Get a single response
            responses = await self._sample_response(messages, n=1)
            if not responses:
                logger.error(
                    f"[Eval Seed: {seed}, Turn: {turn+1}] No response. Aborting."
                )
                break

            llm_output_only = responses[0]
            full_agent_response = agent_prompt_content + llm_output_only

            # Parse and apply the move
            move = self._parse_move(full_agent_response)
            is_valid_move = False
            
            if move is not None:
                is_valid_move = ep.apply_move(move)
                
            if is_valid_move:
                metrics["num_valid_moves"] += 1
            else:
                metrics["num_invalid_moves"] += 1

            # Calculate reward
            is_solved = ep.is_solved()
            reward = self._score_response(is_valid_move, full_agent_response, ep, is_solved)
            metrics["total_reward"] += reward
            metrics["num_turns"] += 1

            # Add response to history
            response_for_history = truncate_thinking(
                full_agent_response, self.tokenizer, self.config.max_think_chars_history
            )
            ep.message_history.append(
                {"role": "agent", "content": response_for_history}
            )

            # Add next observation if not terminal
            is_terminal = is_solved or not is_valid_move
            if not is_terminal:
                ep.message_history.append(
                    {"role": "environment", "content": self._format_observation(ep)}
                )

            # Check for termination
            if is_terminal:
                metrics["is_solved"] = is_solved
                logger.info(f"[Eval Seed: {seed}] Episode ended. Solved: {is_solved}")
                break

        # If we reached max turns, check final state
        if metrics["num_turns"] == max_turns:
            metrics["is_solved"] = ep.is_solved()
            
        # Clean up
        del self.episodes[seed]
        return metrics

    async def evaluate(self, *args, **kwargs):
        """Run evaluation episodes"""
        if not self.config.use_wandb:
            logger.info("Skipping evaluation as wandb is not enabled.")
            return
            
        num_eval_episodes = self.config.eval_episodes
        logger.info(f"Starting evaluation for {num_eval_episodes} episodes.")
        
        eval_tasks = [
            self.rollout_and_score_eval(random.randint(1000001, 2000000))
            for _ in range(num_eval_episodes)
        ]
        
        all_metrics = await tqdm_asyncio.gather(*eval_tasks)
        valid_metrics = [m for m in all_metrics if m]
        
        if not valid_metrics:
            logger.warning("No valid evaluation metrics.")
            return

        # Calculate metrics across all episodes
        num_completed = len(valid_metrics)
        avg_total_reward = sum(m["total_reward"] for m in valid_metrics) / num_completed
        avg_num_turns = sum(m["num_turns"] for m in valid_metrics) / num_completed
        
        total_valid_moves = sum(m["num_valid_moves"] for m in valid_metrics)
        total_invalid_moves = sum(m["num_invalid_moves"] for m in valid_metrics)
        total_moves = total_valid_moves + total_invalid_moves
        move_validity_rate = total_valid_moves / total_moves if total_moves > 0 else 0
        
        solve_rate = sum(1 for m in valid_metrics if m["is_solved"]) / num_completed

        self.eval_metrics = [
            ("eval/avg_total_reward", avg_total_reward),
            ("eval/avg_num_turns", avg_num_turns),
            ("eval/move_validity_rate", move_validity_rate),
            ("eval/solve_rate", solve_rate),
            ("eval/num_completed_episodes", num_completed),
        ]
        
        logger.info(f"Evaluation completed. Metrics: {self.eval_metrics}")

    async def wandb_log(self, wandb_metrics: Optional[Dict[str, float]] = None):
        """Log metrics to wandb"""
        if wandb_metrics is None:
            wandb_metrics = {}
            
        if self.completed_episode_metrics_buffer:
            num_episodes = len(self.completed_episode_metrics_buffer)
            avg_reward = (
                sum(m["total_reward"] for m in self.completed_episode_metrics_buffer)
                / num_episodes
            )
            avg_steps = (
                sum(m["num_steps"] for m in self.completed_episode_metrics_buffer)
                / num_episodes
            )
            solve_rate = (
                sum(
                    1
                    for m in self.completed_episode_metrics_buffer
                    if m["is_solved"]
                )
                / num_episodes
            )
            
            # Log metrics
            wandb_metrics[f"{self.wandb_prepend or 'rubiks'}_train/avg_episode_reward"] = avg_reward
            wandb_metrics[f"{self.wandb_prepend or 'rubiks'}_train/avg_episode_steps"] = avg_steps
            wandb_metrics[f"{self.wandb_prepend or 'rubiks'}_train/solve_rate"] = solve_rate
            wandb_metrics[f"{self.wandb_prepend or 'rubiks'}_train/num_episodes"] = num_episodes
            
            # Clear buffer
            self.completed_episode_metrics_buffer = []
            
        await super().wandb_log(wandb_metrics)

    @classmethod
    def config_init(cls) -> Tuple[RubiksCubeEnvConfig, List[APIServerConfig]]:
        """Initialize the configuration"""
        env_config = RubiksCubeEnvConfig(
            tokenizer_name="openai/gpt-4-turbo-preview",
            group_size=16,
            use_wandb=True,
            max_num_workers=128,
            rollout_server_url="http://localhost:9000",
            total_steps=2000,
            batch_size=1024,
            steps_per_eval=20,
            max_token_length=1024 * 16,
            inference_weight=1.0,
            wandb_name="rubiks_cube",
            data_path_to_save_groups=None,
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.1,
            max_steps=20,
            temperature=0.7,
            top_p=0.9,
            thinking_active=True,
            eval_episodes=100,
            max_think_chars_history=3000,
            max_trajectory_tokens=24576,
            debug_mode=False,
            tiebreak_token_factor=0.01,
            scramble_moves=7,
            cube_size=3,
            reward_per_correctly_placed_cubie=0.05,
            reward_per_step_reduction=0.01,
        )
        
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9004/v1",
                num_requests_for_eval=256,
            )
        ]
        
        return env_config, server_configs

    @classmethod
    def cli(cls):
        """Command-line interface"""
        super().cli()


if __name__ == "__main__":
    RubiksCubeEnv.cli()