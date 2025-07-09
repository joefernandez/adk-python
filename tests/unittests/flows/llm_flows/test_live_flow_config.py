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

from google.adk.flows.llm_flows.audio_cache_manager import AudioCacheConfig
from google.adk.flows.llm_flows.live_flow_config import ControlEventConfig
from google.adk.flows.llm_flows.live_flow_config import LiveFlowConfig


class TestLiveFlowConfig:
  """Test the LiveFlowConfig class."""

  def test_default_values(self):
    """Test that default configuration values are set correctly."""
    config = LiveFlowConfig()

    # Test timing configuration
    assert config.request_queue_timeout == 0.25
    assert config.transfer_agent_delay == 1.0
    assert config.task_completion_delay == 1.0

    # Test feature flags
    assert config.enable_cache_statistics is False

    # Test audio cache config is initialized
    assert config.audio_cache_config is not None
    assert isinstance(config.audio_cache_config, AudioCacheConfig)

  def test_custom_values(self):
    """Test that custom configuration values are set correctly."""
    custom_audio_config = AudioCacheConfig(max_cache_size_bytes=5 * 1024 * 1024)

    config = LiveFlowConfig(
        request_queue_timeout=0.5,
        transfer_agent_delay=2.0,
        task_completion_delay=1.5,
        audio_cache_config=custom_audio_config,
        enable_cache_statistics=True,
    )

    assert config.request_queue_timeout == 0.5
    assert config.transfer_agent_delay == 2.0
    assert config.task_completion_delay == 1.5
    assert config.audio_cache_config == custom_audio_config
    assert config.enable_cache_statistics is True

  def test_audio_cache_config_auto_creation(self):
    """Test that audio cache config is automatically created when None."""
    config = LiveFlowConfig(audio_cache_config=None)

    # Should have created a default AudioCacheConfig
    assert config.audio_cache_config is not None
    assert isinstance(config.audio_cache_config, AudioCacheConfig)
    assert config.audio_cache_config.max_cache_size_bytes == 10 * 1024 * 1024

  def test_partial_configuration(self):
    """Test that partial configuration works correctly."""
    config = LiveFlowConfig(
        request_queue_timeout=0.1, enable_cache_statistics=True
    )

    # Modified values
    assert config.request_queue_timeout == 0.1
    assert config.enable_cache_statistics is True

    # Default values should remain
    assert config.transfer_agent_delay == 1.0
    assert config.audio_cache_config is not None


class TestControlEventConfig:
  """Test the ControlEventConfig class."""

  def test_default_values(self):
    """Test that default control event configuration is correct."""
    config = ControlEventConfig()

    # Test default flush settings
    assert config.flush_on_interrupted == (False, True)
    assert config.flush_on_turn_complete == (True, True)
    assert config.flush_on_generation_complete == (False, True)

  def test_custom_values(self):
    """Test that custom control event configuration is set correctly."""
    config = ControlEventConfig(
        flush_on_interrupted=(True, False),
        flush_on_turn_complete=(False, False),
        flush_on_generation_complete=(True, True),
    )

    assert config.flush_on_interrupted == (True, False)
    assert config.flush_on_turn_complete == (False, False)
    assert config.flush_on_generation_complete == (True, True)

  def test_get_flush_settings_interrupted(self):
    """Test getting flush settings for interrupted events."""
    config = ControlEventConfig(flush_on_interrupted=(True, False))

    settings = config.get_flush_settings('interrupted')
    assert settings == (True, False)

  def test_get_flush_settings_turn_complete(self):
    """Test getting flush settings for turn complete events."""
    config = ControlEventConfig(flush_on_turn_complete=(False, True))

    settings = config.get_flush_settings('turn_complete')
    assert settings == (False, True)

  def test_get_flush_settings_generation_complete(self):
    """Test getting flush settings for generation complete events."""
    config = ControlEventConfig(flush_on_generation_complete=(True, True))

    settings = config.get_flush_settings('generation_complete')
    assert settings == (True, True)

  def test_get_flush_settings_unknown_event(self):
    """Test getting flush settings for unknown event types."""
    config = ControlEventConfig()

    settings = config.get_flush_settings('unknown_event')
    assert settings == (False, False)

  def test_get_flush_settings_none_event(self):
    """Test getting flush settings for None event type."""
    config = ControlEventConfig()

    settings = config.get_flush_settings(None)
    assert settings == (False, False)

  def test_get_flush_settings_empty_string(self):
    """Test getting flush settings for empty string event type."""
    config = ControlEventConfig()

    settings = config.get_flush_settings('')
    assert settings == (False, False)

  def test_all_event_types(self):
    """Test all supported event types with different configurations."""
    config = ControlEventConfig(
        flush_on_interrupted=(True, False),
        flush_on_turn_complete=(False, True),
        flush_on_generation_complete=(True, True),
    )

    # Test all event types
    assert config.get_flush_settings('interrupted') == (True, False)
    assert config.get_flush_settings('turn_complete') == (False, True)
    assert config.get_flush_settings('generation_complete') == (True, True)

  def test_boolean_combinations(self):
    """Test all possible boolean combinations for flush settings."""
    test_cases = [(False, False), (False, True), (True, False), (True, True)]

    for user_audio, model_audio in test_cases:
      config = ControlEventConfig(
          flush_on_interrupted=(user_audio, model_audio)
      )
      settings = config.get_flush_settings('interrupted')
      assert settings == (user_audio, model_audio)
