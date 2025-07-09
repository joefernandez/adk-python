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

import asyncio
import time

from google.adk.agents import Agent
from google.adk.agents import LiveRequestQueue
from google.adk.agents.invocation_context import AudioCacheEntry
from google.adk.agents.run_config import RunConfig
from google.adk.events.event import Event
from google.adk.models import LlmResponse
from google.genai import types
import pytest

from .. import testing_utils


def test_audio_caching_direct():
  """Test audio caching logic directly without full live streaming."""
  # This test directly verifies that our audio caching logic works
  audio_data = b'\x00\xFF\x01\x02\x03\x04\x05\x06'
  audio_mime_type = 'audio/pcm'

  # Create mock responses for successful completion
  responses = [
      LlmResponse(
          content=types.Content(
              role='model',
              parts=[types.Part.from_text(text='Processing audio...')],
          ),
          turn_complete=False,
      ),
      LlmResponse(turn_complete=True),  # This should trigger flush
  ]

  mock_model = testing_utils.MockModel.create(responses)
  mock_model.model = 'gemini-2.0-flash-exp'  # For CFC support

  root_agent = Agent(
      name='test_agent',
      model=mock_model,
      tools=[],
  )

  # Test our implementation by directly calling it
  async def test_caching():
    # Create context similar to what would be created in real scenario
    invocation_context = await testing_utils.create_invocation_context(
        root_agent, run_config=RunConfig(support_cfc=True)
    )

    # Import our caching classes
    from google.adk.agents.invocation_context import AudioCacheEntry
    from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow

    # Create a mock flow to test our methods
    flow = BaseLlmFlow()

    # Test adding audio to cache
    invocation_context.input_audio_cache = []
    audio_entry = AudioCacheEntry(
        role='user',
        data=types.Blob(data=audio_data, mime_type=audio_mime_type),
        timestamp=1234567890.0,
    )
    invocation_context.input_audio_cache.append(audio_entry)

    # Verify cache has data
    assert len(invocation_context.input_audio_cache) == 1
    assert invocation_context.input_audio_cache[0].data.data == audio_data

    # Test flushing cache
    await flow._flush_audio_caches(invocation_context)

    # Verify cache was cleared
    assert len(invocation_context.input_audio_cache) == 0

    # Check if artifacts were created
    artifact_keys = (
        await invocation_context.artifact_service.list_artifact_keys(
            app_name=invocation_context.app_name,
            user_id=invocation_context.user_id,
            session_id=invocation_context.session.id,
        )
    )

    # Should have at least one audio artifact
    audio_artifacts = [key for key in artifact_keys if 'audio' in key.lower()]
    assert (
        len(audio_artifacts) > 0
    ), f'Expected audio artifacts, found: {artifact_keys}'

    # Verify artifact content
    if audio_artifacts:
      artifact = await invocation_context.artifact_service.load_artifact(
          app_name=invocation_context.app_name,
          user_id=invocation_context.user_id,
          session_id=invocation_context.session.id,
          filename=audio_artifacts[0],
      )
      assert artifact.inline_data.data == audio_data

    return True

  # Run the async test
  result = asyncio.run(test_caching())
  assert result is True


def test_transcription_handling():
  """Test that transcriptions are properly handled and saved to session service."""

  # Create mock responses with transcriptions
  input_transcription = types.Transcription(
      text='Hello, this is transcribed input', finished=True
  )
  output_transcription = types.Transcription(
      text='This is transcribed output', finished=True
  )

  responses = [
      LlmResponse(
          content=types.Content(
              role='model', parts=[types.Part.from_text(text='Processing...')]
          ),
          turn_complete=False,
      ),
      LlmResponse(input_transcription=input_transcription, turn_complete=False),
      LlmResponse(
          output_transcription=output_transcription, turn_complete=False
      ),
      LlmResponse(turn_complete=True),
  ]

  mock_model = testing_utils.MockModel.create(responses)
  mock_model.model = 'gemini-2.0-flash-exp'

  root_agent = Agent(
      name='test_agent',
      model=mock_model,
      tools=[],
  )

  async def test_transcription():
    # Create context
    invocation_context = await testing_utils.create_invocation_context(
        root_agent, run_config=RunConfig(support_cfc=True)
    )

    from google.adk.events.event import Event
    from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow

    flow = BaseLlmFlow()

    # Test processing transcription events
    session_events_before = len(invocation_context.session.events)

    # Simulate input transcription event
    input_event = Event(
        id=Event.new_id(),
        invocation_id=invocation_context.invocation_id,
        author='user',
        input_transcription=input_transcription,
    )

    # Simulate output transcription event
    output_event = Event(
        id=Event.new_id(),
        invocation_id=invocation_context.invocation_id,
        author=invocation_context.agent.name,
        output_transcription=output_transcription,
    )

    # Save transcription events to session
    await invocation_context.session_service.append_event(
        invocation_context.session, input_event
    )
    await invocation_context.session_service.append_event(
        invocation_context.session, output_event
    )

    # Verify transcriptions were saved to session
    session_events_after = len(invocation_context.session.events)
    assert session_events_after == session_events_before + 2

    # Check that transcription events were saved
    transcription_events = [
        event
        for event in invocation_context.session.events
        if hasattr(event, 'input_transcription')
        and event.input_transcription
        or hasattr(event, 'output_transcription')
        and event.output_transcription
    ]
    assert len(transcription_events) >= 2

    # Verify input transcription
    input_transcription_events = [
        event
        for event in invocation_context.session.events
        if hasattr(event, 'input_transcription') and event.input_transcription
    ]
    assert len(input_transcription_events) >= 1
    assert (
        input_transcription_events[0].input_transcription.text
        == 'Hello, this is transcribed input'
    )
    assert input_transcription_events[0].author == 'user'

    # Verify output transcription
    output_transcription_events = [
        event
        for event in invocation_context.session.events
        if hasattr(event, 'output_transcription') and event.output_transcription
    ]
    assert len(output_transcription_events) >= 1
    assert (
        output_transcription_events[0].output_transcription.text
        == 'This is transcribed output'
    )
    assert (
        output_transcription_events[0].author == invocation_context.agent.name
    )

    return True

  # Run the async test
  result = asyncio.run(test_transcription())
  assert result is True


def test_live_streaming_with_transcriptions():
  """Test transcriptions within the actual live streaming flow."""

  # Create mock responses including transcriptions and audio
  audio_data = b'\x00\xFF\x01\x02\x03\x04\x05\x06'
  input_transcription = types.Transcription(
      text='User said hello', finished=True
  )
  output_transcription = types.Transcription(
      text='Model response transcription', finished=True
  )

  responses = [
      LlmResponse(
          content=types.Content(
              role='model',
              parts=[types.Part.from_text(text='Starting conversation...')],
          ),
          turn_complete=False,
      ),
      LlmResponse(input_transcription=input_transcription, turn_complete=False),
      LlmResponse(
          content=types.Content(
              role='model',
              parts=[
                  types.Part(
                      inline_data=types.Blob(
                          data=audio_data, mime_type='audio/pcm'
                      )
                  )
              ],
          ),
          turn_complete=False,
      ),
      LlmResponse(
          output_transcription=output_transcription, turn_complete=False
      ),
      LlmResponse(turn_complete=True),
  ]

  mock_model = testing_utils.MockModel.create(responses)
  mock_model.model = 'gemini-2.0-flash-exp'

  root_agent = Agent(
      name='test_agent',
      model=mock_model,
      tools=[],
  )

  async def test_live_flow():
    # Create context for live streaming
    invocation_context = await testing_utils.create_invocation_context(
        root_agent, run_config=RunConfig(support_cfc=True)
    )

    from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow

    flow = BaseLlmFlow()
    session_events_before = len(invocation_context.session.events)

    # Mock the llm connection for the test
    class MockLlmConnection:

      def __init__(self, responses):
        self.responses = responses
        self.index = 0

      async def send_history(self, history):
        pass

      async def receive(self):
        for response in self.responses:
          yield response

      async def close(self):
        pass

    # Create mock connection
    mock_connection = MockLlmConnection(responses)

    # Simulate processing responses through our flow
    event_count = 0
    transcription_count = 0
    audio_cache_flushed = False

    async for llm_response in mock_connection.receive():
      # Simulate what _receive_from_model does
      model_response_event = Event(
          id=Event.new_id(),
          invocation_id=invocation_context.invocation_id,
          author=invocation_context.agent.name,
      )

      # Process transcriptions
      if llm_response.input_transcription:
        await flow._handle_transcription_event(
            invocation_context, llm_response.input_transcription, 'user'
        )
        transcription_count += 1

      if llm_response.output_transcription:
        await flow._handle_transcription_event(
            invocation_context,
            llm_response.output_transcription,
            invocation_context.agent.name,
        )
        transcription_count += 1

      # Process audio caching
      if (
          llm_response.content
          and llm_response.content.parts
          and llm_response.content.parts[0].inline_data
          and llm_response.content.parts[0].inline_data.mime_type.startswith(
              'audio/'
          )
      ):
        if not invocation_context.output_audio_cache:
          invocation_context.output_audio_cache = []
        invocation_context.output_audio_cache.append(
            AudioCacheEntry(
                role='model',
                data=types.Blob(
                    data=llm_response.content.parts[0].inline_data.data,
                    mime_type=llm_response.content.parts[
                        0
                    ].inline_data.mime_type,
                ),
                timestamp=time.time(),
            )
        )

      # Flush caches on turn complete
      if llm_response.turn_complete:
        await flow._flush_audio_caches(invocation_context)
        audio_cache_flushed = True

      event_count += 1

    # Verify results
    session_events_after = len(invocation_context.session.events)

    # Should have added transcription events
    assert (
        transcription_count == 2
    ), f'Expected 2 transcriptions, got {transcription_count}'

    # Session should have the transcription events
    transcription_events = [
        event
        for event in invocation_context.session.events
        if (hasattr(event, 'input_transcription') and event.input_transcription)
        or (
            hasattr(event, 'output_transcription')
            and event.output_transcription
        )
    ]
    assert len(transcription_events) >= 2

    # Verify input transcription event
    input_events = [e for e in transcription_events if e.input_transcription]
    assert len(input_events) >= 1
    assert input_events[0].input_transcription.text == 'User said hello'
    assert input_events[0].author == 'user'

    # Verify output transcription event
    output_events = [e for e in transcription_events if e.output_transcription]
    assert len(output_events) >= 1
    assert (
        output_events[0].output_transcription.text
        == 'Model response transcription'
    )
    assert output_events[0].author == invocation_context.agent.name

    # Audio cache should have been flushed
    assert audio_cache_flushed
    assert len(invocation_context.output_audio_cache) == 0

    return True

  # Run the async test
  result = asyncio.run(test_live_flow())
  assert result is True


def test_selective_audio_cache_flushing():
  """Test that different control events flush the correct audio caches."""

  mock_model = testing_utils.MockModel.create([])
  mock_model.model = 'gemini-2.0-flash-exp'

  root_agent = Agent(
      name='test_agent',
      model=mock_model,
      tools=[],
  )

  async def test_flushing_logic():
    # Create context
    invocation_context = await testing_utils.create_invocation_context(
        root_agent, run_config=RunConfig(support_cfc=True)
    )

    from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow

    flow = BaseLlmFlow()

    # Add some mock audio data to both caches
    audio_data = b'\x00\xFF\x01\x02'
    invocation_context.input_audio_cache = [
        AudioCacheEntry(
            role='user',
            data=types.Blob(data=audio_data, mime_type='audio/pcm'),
            timestamp=time.time(),
        )
    ]
    invocation_context.output_audio_cache = [
        AudioCacheEntry(
            role='model',
            data=types.Blob(data=audio_data, mime_type='audio/pcm'),
            timestamp=time.time(),
        )
    ]

    # Test 1: interrupted event should flush model audio only
    await flow._flush_audio_caches(
        invocation_context, flush_user_audio=False, flush_model_audio=True
    )

    # User audio should remain, model audio should be flushed
    assert (
        len(invocation_context.input_audio_cache) == 1
    ), 'User audio should not be flushed on interrupted'
    assert (
        len(invocation_context.output_audio_cache) == 0
    ), 'Model audio should be flushed on interrupted'

    # Reset for next test
    invocation_context.output_audio_cache = [
        AudioCacheEntry(
            role='model',
            data=types.Blob(data=audio_data, mime_type='audio/pcm'),
            timestamp=time.time(),
        )
    ]

    # Test 2: turn_complete should flush both user and model audio
    await flow._flush_audio_caches(
        invocation_context, flush_user_audio=True, flush_model_audio=True
    )

    # Both caches should be empty
    assert (
        len(invocation_context.input_audio_cache) == 0
    ), 'User audio should be flushed on turn_complete'
    assert (
        len(invocation_context.output_audio_cache) == 0
    ), 'Model audio should be flushed on turn_complete'

    # Reset for next test
    invocation_context.input_audio_cache = [
        AudioCacheEntry(
            role='user',
            data=types.Blob(data=audio_data, mime_type='audio/pcm'),
            timestamp=time.time(),
        )
    ]
    invocation_context.output_audio_cache = [
        AudioCacheEntry(
            role='model',
            data=types.Blob(data=audio_data, mime_type='audio/pcm'),
            timestamp=time.time(),
        )
    ]

    # Test 3: generation_complete should flush model audio only
    await flow._flush_audio_caches(
        invocation_context, flush_user_audio=False, flush_model_audio=True
    )

    # User audio should remain, model audio should be flushed
    assert (
        len(invocation_context.input_audio_cache) == 1
    ), 'User audio should not be flushed on generation_complete'
    assert (
        len(invocation_context.output_audio_cache) == 0
    ), 'Model audio should be flushed on generation_complete'

    return True

  # Run the async test
  result = asyncio.run(test_flushing_logic())
  assert result is True
