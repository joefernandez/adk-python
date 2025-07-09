# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

from google.adk.flows.llm_flows.audio_cache_manager import AudioCacheConfig
from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow
from google.adk.flows.llm_flows.live_flow_config import ControlEventConfig
from google.adk.flows.llm_flows.live_flow_config import LiveFlowConfig
from google.adk.models.llm_response import LlmResponse
from google.genai import types
import pytest

from ... import testing_utils


class TestBaseLlmFlowRefactored:
  """Test the refactored BaseLlmFlow class with new manager classes."""

  def setup_method(self):
    """Set up test fixtures."""
    self.live_flow_config = LiveFlowConfig(
        request_queue_timeout=0.1, enable_cache_statistics=True
    )
    self.control_event_config = ControlEventConfig()

    # Create a concrete implementation for testing
    class TestLlmFlow(BaseLlmFlow):

      async def _run_async_impl(self, ctx):
        # Minimal implementation for testing
        pass

      async def _run_live_impl(self, ctx):
        # Minimal implementation for testing
        pass

    self.flow = TestLlmFlow(self.live_flow_config, self.control_event_config)

  def test_initialization_with_config(self):
    """Test that the flow initializes correctly with configuration."""
    assert self.flow.live_flow_config == self.live_flow_config
    assert self.flow.control_event_config == self.control_event_config
    assert self.flow.audio_cache_manager is not None
    assert self.flow.transcription_manager is not None

  def test_initialization_with_defaults(self):
    """Test that the flow initializes correctly with default configuration."""

    class TestLlmFlow(BaseLlmFlow):

      async def _run_async_impl(self, ctx):
        pass

      async def _run_live_impl(self, ctx):
        pass

    flow = TestLlmFlow()

    assert flow.live_flow_config is not None
    assert flow.control_event_config is not None
    assert flow.audio_cache_manager is not None
    assert flow.transcription_manager is not None

    # Check default values
    assert flow.live_flow_config.enable_cache_statistics is False

  @pytest.mark.asyncio
  async def test_handle_control_event_flush_interrupted(self):
    """Test handling interrupted control events for cache flushing."""
    invocation_context = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )

    # Set up mock artifact service
    mock_artifact_service = AsyncMock()
    invocation_context.artifact_service = mock_artifact_service

    # Cache some audio first
    audio_blob = types.Blob(data=b'test_data', mime_type='audio/pcm')
    self.flow.audio_cache_manager.cache_input_audio(
        invocation_context, audio_blob
    )
    self.flow.audio_cache_manager.cache_output_audio(
        invocation_context, audio_blob
    )

    # Create LLM response with interrupted flag
    llm_response = LlmResponse(interrupted=True)

    # Handle control event flush
    await self.flow._handle_control_event_flush(
        invocation_context, llm_response
    )

    # Check that the correct caches were flushed based on config
    # Default interrupted config is (False, True) - only model audio
    assert (
        len(invocation_context.input_audio_cache) == 1
    )  # User audio not flushed
    assert (
        len(invocation_context.output_audio_cache) == 0
    )  # Model audio flushed

  @pytest.mark.asyncio
  async def test_handle_control_event_flush_turn_complete(self):
    """Test handling turn complete control events for cache flushing."""
    invocation_context = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )

    # Set up mock artifact service
    mock_artifact_service = AsyncMock()
    invocation_context.artifact_service = mock_artifact_service

    # Cache some audio first
    audio_blob = types.Blob(data=b'test_data', mime_type='audio/pcm')
    self.flow.audio_cache_manager.cache_input_audio(
        invocation_context, audio_blob
    )
    self.flow.audio_cache_manager.cache_output_audio(
        invocation_context, audio_blob
    )

    # Create LLM response with turn_complete flag
    llm_response = LlmResponse(turn_complete=True)

    # Handle control event flush
    await self.flow._handle_control_event_flush(
        invocation_context, llm_response
    )

    # Check that both caches were flushed based on config
    # Default turn_complete config is (True, True) - both caches
    assert len(invocation_context.input_audio_cache) == 0  # User audio flushed
    assert (
        len(invocation_context.output_audio_cache) == 0
    )  # Model audio flushed

  @pytest.mark.asyncio
  async def test_handle_control_event_flush_generation_complete(self):
    """Test handling generation complete control events for cache flushing."""
    invocation_context = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )

    # Set up mock artifact service
    mock_artifact_service = AsyncMock()
    invocation_context.artifact_service = mock_artifact_service

    # Cache some audio first
    audio_blob = types.Blob(data=b'test_data', mime_type='audio/pcm')
    self.flow.audio_cache_manager.cache_input_audio(
        invocation_context, audio_blob
    )
    self.flow.audio_cache_manager.cache_output_audio(
        invocation_context, audio_blob
    )

    # Create mock LLM response with generation_complete flag
    llm_response = Mock()
    llm_response.interrupted = None
    llm_response.turn_complete = None
    llm_response.generation_complete = True

    # Handle control event flush
    await self.flow._handle_control_event_flush(
        invocation_context, llm_response
    )

    # Check that the correct caches were flushed based on config
    # Default generation_complete config is (False, True) - only model audio
    assert (
        len(invocation_context.input_audio_cache) == 1
    )  # User audio not flushed
    assert (
        len(invocation_context.output_audio_cache) == 0
    )  # Model audio flushed

  @pytest.mark.asyncio
  async def test_custom_control_event_config(self):
    """Test using custom control event configuration."""
    # Create custom config that flushes everything on interrupted
    custom_control_config = ControlEventConfig(
        flush_on_interrupted=(True, True)
    )

    class TestLlmFlow(BaseLlmFlow):

      async def _run_async_impl(self, ctx):
        pass

      async def _run_live_impl(self, ctx):
        pass

    flow = TestLlmFlow(control_event_config=custom_control_config)

    invocation_context = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )

    # Set up mock artifact service
    mock_artifact_service = AsyncMock()
    invocation_context.artifact_service = mock_artifact_service

    # Cache some audio first
    audio_blob = types.Blob(data=b'test_data', mime_type='audio/pcm')
    flow.audio_cache_manager.cache_input_audio(invocation_context, audio_blob)
    flow.audio_cache_manager.cache_output_audio(invocation_context, audio_blob)

    # Create LLM response with interrupted flag
    llm_response = LlmResponse(interrupted=True)

    # Handle control event flush
    await flow._handle_control_event_flush(invocation_context, llm_response)

    # Check that both caches were flushed with custom config
    assert len(invocation_context.input_audio_cache) == 0  # User audio flushed
    assert (
        len(invocation_context.output_audio_cache) == 0
    )  # Model audio flushed

  def test_configuration_inheritance(self):
    """Test that configuration is properly passed to managers."""
    custom_audio_config = AudioCacheConfig(
        max_cache_size_bytes=1024, auto_flush_threshold=10
    )

    config = LiveFlowConfig(
        audio_cache_config=custom_audio_config, request_queue_timeout=0.5
    )

    class TestLlmFlow(BaseLlmFlow):

      async def _run_async_impl(self, ctx):
        pass

      async def _run_live_impl(self, ctx):
        pass

    flow = TestLlmFlow(config)

    # Verify audio cache manager received the custom config
    assert flow.audio_cache_manager.config == custom_audio_config
    assert flow.audio_cache_manager.config.max_cache_size_bytes == 1024
    assert flow.audio_cache_manager.config.auto_flush_threshold == 10

  @pytest.mark.asyncio
  async def test_statistics_logging(self):
    """Test that statistics logging works when enabled."""
    config = LiveFlowConfig(enable_cache_statistics=True)

    class TestLlmFlow(BaseLlmFlow):

      async def _run_async_impl(self, ctx):
        pass

      async def _run_live_impl(self, ctx):
        pass

    flow = TestLlmFlow(config)

    invocation_context = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )

    # Cache some audio to have stats to log
    audio_blob = types.Blob(data=b'test_data', mime_type='audio/pcm')
    flow.audio_cache_manager.cache_input_audio(invocation_context, audio_blob)

    # Mock the logger to verify it gets called
    with patch(
        'google.adk.flows.llm_flows.base_llm_flow.logger'
    ) as mock_logger:
      llm_response = LlmResponse(turn_complete=True)

      # Set up mock artifact service
      mock_artifact_service = AsyncMock()
      invocation_context.artifact_service = mock_artifact_service

      await flow._handle_control_event_flush(invocation_context, llm_response)

      # Verify that debug logging was called for statistics
      mock_logger.debug.assert_called()
      # Check that the call was for cache stats
      debug_calls = [
          call
          for call in mock_logger.debug.call_args_list
          if 'Audio cache stats:' in str(call)
      ]
      assert len(debug_calls) > 0

  @pytest.mark.asyncio
  async def test_no_statistics_logging_when_disabled(self):
    """Test that statistics logging doesn't happen when disabled."""
    config = LiveFlowConfig(enable_cache_statistics=False)

    class TestLlmFlow(BaseLlmFlow):

      async def _run_async_impl(self, ctx):
        pass

      async def _run_live_impl(self, ctx):
        pass

    flow = TestLlmFlow(config)

    invocation_context = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )

    # Mock the logger to verify it doesn't get called for stats
    with patch(
        'google.adk.flows.llm_flows.base_llm_flow.logger'
    ) as mock_logger:
      llm_response = LlmResponse(turn_complete=True)

      await flow._handle_control_event_flush(invocation_context, llm_response)

      # Verify that no debug logging was called for statistics
      debug_calls = [
          call
          for call in mock_logger.debug.call_args_list
          if 'Audio cache stats:' in str(call)
      ]
      assert len(debug_calls) == 0
