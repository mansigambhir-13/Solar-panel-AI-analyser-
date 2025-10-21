#!/usr/bin/env python3
"""
enhanced_decision_agent.py
Advanced Decision Orchestration Agent for Solar Panel Cleaning System
Enhanced with Text-to-Speech capabilities and improved features

Author: AI System
Version: 3.0.0
Date: 2024
"""

import os
import time
import json
import numpy as np
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass
from enum import Enum

# TTS and Audio imports
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("Warning: pyttsx3 not available. Install with: pip install pyttsx3")

try:
    import azure.cognitiveservices.speech as speechsdk
    AZURE_TTS_AVAILABLE = True
except ImportError:
    AZURE_TTS_AVAILABLE = False

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# Configure logging with improved formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('solar_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TTSEngine(Enum):
    """TTS Engine options"""
    PYTTSX3 = "pyttsx3"
    AZURE = "azure"
    ESPEAK = "espeak"
    FESTIVAL = "festival"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    SUCCESS = "success"


@dataclass
class TTSConfig:
    """TTS Configuration settings"""
    engine: TTSEngine = TTSEngine.PYTTSX3
    rate: int = 150  # Words per minute
    volume: float = 0.9  # 0.0 to 1.0
    voice_index: int = 0  # Voice selection
    enabled: bool = True
    azure_key: Optional[str] = None
    azure_region: Optional[str] = None
    language: str = "en-US"
    voice_name: str = "en-US-JennyNeural"  # Azure voice
    audio_alerts: bool = True  # Play audio alerts for different states


class TTSManager:
    """
    Text-to-Speech Manager with multiple engine support
    Provides voice feedback for critical system events and decisions
    """
    
    def __init__(self, config: TTSConfig = None):
        self.config = config or TTSConfig()
        self.engine = None
        self.is_speaking = False
        self.speech_queue = []
        
        # Initialize pygame for audio alerts if available
        if PYGAME_AVAILABLE and self.config.audio_alerts:
            try:
                pygame.mixer.init()
                self.audio_enabled = True
            except:
                self.audio_enabled = False
                logger.warning("Audio alerts disabled - pygame initialization failed")
        else:
            self.audio_enabled = False
        
        self._initialize_tts_engine()
    
    def _initialize_tts_engine(self):
        """Initialize the selected TTS engine"""
        if not self.config.enabled:
            logger.info("TTS disabled in configuration")
            return
        
        try:
            if self.config.engine == TTSEngine.PYTTSX3 and TTS_AVAILABLE:
                self.engine = pyttsx3.init()
                self._configure_pyttsx3()
                logger.info("✅ pyttsx3 TTS engine initialized")
                
            elif self.config.engine == TTSEngine.AZURE and AZURE_TTS_AVAILABLE:
                self._configure_azure_tts()
                logger.info("✅ Azure TTS engine initialized")
                
            elif self.config.engine == TTSEngine.ESPEAK:
                # Test if espeak is available
                if os.system("which espeak > /dev/null 2>&1") == 0:
                    self.engine = "espeak"
                    logger.info("✅ eSpeak TTS engine initialized")
                else:
                    raise Exception("eSpeak not found in system")
                    
            else:
                logger.warning("No suitable TTS engine available, using fallback")
                self.config.enabled = False
                
        except Exception as e:
            logger.error(f"TTS initialization failed: {e}")
            self.config.enabled = False
    
    def _configure_pyttsx3(self):
        """Configure pyttsx3 engine settings"""
        if self.engine:
            # Set speech rate
            self.engine.setProperty('rate', self.config.rate)
            
            # Set volume
            self.engine.setProperty('volume', self.config.volume)
            
            # Set voice
            voices = self.engine.getProperty('voices')
            if voices and len(voices) > self.config.voice_index:
                self.engine.setProperty('voice', voices[self.config.voice_index].id)
    
    def _configure_azure_tts(self):
        """Configure Azure TTS settings"""
        if not self.config.azure_key or not self.config.azure_region:
            raise Exception("Azure TTS requires API key and region")
        
        speech_config = speechsdk.SpeechConfig(
            subscription=self.config.azure_key,
            region=self.config.azure_region
        )
        speech_config.speech_synthesis_voice_name = self.config.voice_name
        self.azure_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
    
    def speak(self, text: str, priority: bool = False, alert_level: AlertLevel = AlertLevel.INFO):
        """
        Convert text to speech with priority and alert level support
        
        Args:
            text: Text to speak
            priority: If True, interrupt current speech
            alert_level: Alert level for audio cues
        """
        if not self.config.enabled or not text.strip():
            return
        
        # Add audio alert before speech if enabled
        if self.audio_enabled and alert_level != AlertLevel.INFO:
            self._play_alert_sound(alert_level)
        
        if priority and self.is_speaking:
            self.stop_speech()
        
        try:
            if self.config.engine == TTSEngine.PYTTSX3 and self.engine:
                self._speak_pyttsx3(text)
                
            elif self.config.engine == TTSEngine.AZURE:
                self._speak_azure(text)
                
            elif self.config.engine == TTSEngine.ESPEAK:
                self._speak_espeak(text)
                
        except Exception as e:
            logger.error(f"TTS speech failed: {e}")
    
    def _speak_pyttsx3(self, text: str):
        """Speak using pyttsx3"""
        self.is_speaking = True
        self.engine.say(text)
        self.engine.runAndWait()
        self.is_speaking = False
    
    def _speak_azure(self, text: str):
        """Speak using Azure TTS"""
        self.is_speaking = True
        result = self.azure_synthesizer.speak_text_async(text).get()
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            logger.info("Azure TTS synthesis completed")
        else:
            logger.error(f"Azure TTS failed: {result.reason}")
        self.is_speaking = False
    
    def _speak_espeak(self, text: str):
        """Speak using eSpeak"""
        self.is_speaking = True
        escaped_text = text.replace('"', '\\"')
        os.system(f'espeak -s {self.config.rate} -a {int(self.config.volume * 200)} "{escaped_text}"')
        self.is_speaking = False
    
    def _play_alert_sound(self, alert_level: AlertLevel):
        """Play audio alert based on alert level"""
        if not self.audio_enabled:
            return
        
        # You can add custom sound files here
        alert_sounds = {
            AlertLevel.WARNING: "warning_beep",
            AlertLevel.CRITICAL: "critical_alarm", 
            AlertLevel.SUCCESS: "success_chime"
        }
        
        # For now, use system beeps (you can replace with audio files)
        try:
            if alert_level == AlertLevel.CRITICAL:
                # Critical: 3 quick beeps
                for _ in range(3):
                    print("\a", end="", flush=True)  # System beep
                    time.sleep(0.2)
            elif alert_level == AlertLevel.WARNING:
                # Warning: 2 beeps
                for _ in range(2):
                    print("\a", end="", flush=True)
                    time.sleep(0.3)
            elif alert_level == AlertLevel.SUCCESS:
                # Success: 1 pleasant beep
                print("\a", end="", flush=True)
        except:
            pass
    
    def stop_speech(self):
        """Stop current speech"""
        if self.config.engine == TTSEngine.PYTTSX3 and self.engine:
            self.engine.stop()
        self.is_speaking = False
    
    def announce_decision(self, decision: Dict[str, Any]):
        """Announce decision outcome with appropriate urgency"""
        execute_cleaning = decision.get('execute_cleaning', False)
        confidence = decision.get('decision_confidence', 0)
        reasoning = decision.get('primary_reasoning', 'Unknown reason')
        
        if execute_cleaning:
            if confidence >= 90:
                alert_level = AlertLevel.CRITICAL
                announcement = f"Critical cleaning action required. Confidence {confidence:.0f} percent. {reasoning}"
            else:
                alert_level = AlertLevel.WARNING
                announcement = f"Cleaning recommended. Confidence {confidence:.0f} percent. {reasoning}"
        else:
            alert_level = AlertLevel.INFO
            announcement = f"No cleaning action needed. Confidence {confidence:.0f} percent. {reasoning}"
        
        self.speak(announcement, priority=True, alert_level=alert_level)
    
    def announce_execution_result(self, execution_result: Dict[str, Any]):
        """Announce execution results"""
        if execution_result.get('execution_successful'):
            water_used = execution_result.get('water_usage_actual_liters', 0)
            announcement = f"Cleaning successfully completed. Water used: {water_used:.1f} liters."
            self.speak(announcement, alert_level=AlertLevel.SUCCESS)
        else:
            failure_reason = execution_result.get('failure_reason', 'unknown error')
            announcement = f"Cleaning execution failed due to {failure_reason}. Please check system status."
            self.speak(announcement, priority=True, alert_level=AlertLevel.CRITICAL)


class EnhancedDecisionOrchestrationAgent:
    """
    Enhanced Decision Orchestration Agent with TTS capabilities and improvements
    """
    
    def __init__(self, config: Optional[Dict] = None, tts_config: Optional[TTSConfig] = None):
        """Initialize the enhanced agent with TTS support"""
        self.agent_name = "Enhanced Decision Orchestration Agent"
        self.version = "3.0.0"
        self.config = config or {}
        
        # Initialize TTS Manager
        self.tts = TTSManager(tts_config or TTSConfig())
        
        # Enhanced decision framework with voice feedback
        self.decision_framework = {
            'dust_severity_weight': 0.25,
            'economic_impact_weight': 0.25,
            'risk_assessment_weight': 0.20,
            'confidence_weight': 0.15,
            'timing_optimization_weight': 0.15
        }
        
        # Decision thresholds
        self.decision_thresholds = {
            'critical_action_score': 85,
            'recommended_action_score': 70,
            'optional_action_score': 50,
            'minimum_confidence': 60,
            'maximum_acceptable_risk': 80
        }
        
        # Enhanced spray system with voice status updates
        self.spray_system = {
            'gpio_pin': 18,
            'enable_hardware': os.environ.get('ENABLE_GPIO', 'false').lower() == 'true',
            'simulation_mode': True,
            'safety_protocols_enabled': True,
            'voice_feedback_enabled': True
        }
        
        # Economic parameters
        self.economic_params = {
            'electricity_rate_per_kwh': 0.12,
            'water_cost_per_liter': 0.05,
            'labor_cost_per_hour': 15.0,
            'equipment_cost_per_use': 3.0,
            'carbon_cost_per_kg': 0.05
        }
        
        # Callback system for external integrations
        self.callbacks = {
            'on_decision_made': [],
            'on_execution_complete': [],
            'on_critical_alert': [],
            'on_system_error': []
        }
        
        # Performance metrics tracking
        self.performance_metrics = {
            'total_decisions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'average_confidence': 0,
            'uptime_start': datetime.now()
        }
        
        self._initialize_enhanced_systems()
        
        # Welcome announcement
        self.tts.speak(f"{self.agent_name} version {self.version} initialized successfully", 
                       alert_level=AlertLevel.SUCCESS)
        
        logger.info(f"✅ {self.agent_name} v{self.version} initialized with TTS support")
    
    def _initialize_enhanced_systems(self):
        """Initialize enhanced system components"""
        # Initialize spray system with voice feedback
        if self.spray_system['enable_hardware']:
            try:
                # Hardware initialization would go here
                logger.info("🚿 Spray system: Hardware GPIO enabled")
                self.spray_system['simulation_mode'] = False
                if self.spray_system['voice_feedback_enabled']:
                    self.tts.speak("Hardware spray system enabled", alert_level=AlertLevel.INFO)
            except Exception as e:
                logger.warning(f"GPIO initialization failed, using simulation: {e}")
                self.spray_system['simulation_mode'] = True
                self.tts.speak("Hardware initialization failed, using simulation mode", 
                              alert_level=AlertLevel.WARNING)
        else:
            logger.info("🚿 Spray system: Simulation mode")
            self.tts.speak("Spray system running in simulation mode")
    
    def register_callback(self, event: str, callback: Callable):
        """Register callback for system events"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
            logger.info(f"Callback registered for event: {event}")
    
    def _trigger_callbacks(self, event: str, data: Any):
        """Trigger registered callbacks for an event"""
        for callback in self.callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback execution failed for {event}: {e}")
    
    async def orchestrate_comprehensive_decision_async(self, dust_result: Dict[str, Any],
                                                      forecast_result: Dict[str, Any],
                                                      image_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Async version of comprehensive decision orchestration with voice updates
        """
        logger.info("🎯 Decision Orchestration: Starting comprehensive async analysis...")
        self.tts.speak("Starting comprehensive system analysis", alert_level=AlertLevel.INFO)
        
        start_time = time.time()
        
        try:
            # Phase 1: Data integration with voice update
            logger.info("📊 Phase 1: Data integration and validation")
            self.tts.speak("Phase one: Data integration and validation")
            data_integration = await self._integrate_and_validate_inputs_async(
                dust_result, forecast_result, image_result
            )
            
            # Phase 2: Risk assessment with voice update
            logger.info("⚠️ Phase 2: Multi-dimensional risk assessment")
            self.tts.speak("Phase two: Risk assessment in progress")
            risk_assessment = self._perform_multidimensional_risk_assessment(data_integration)
            
            # Announce critical risks immediately
            if risk_assessment['overall_risk_assessment']['overall_risk_score'] > 85:
                self.tts.speak("Critical risk level detected. Immediate attention required.", 
                              priority=True, alert_level=AlertLevel.CRITICAL)
                self._trigger_callbacks('on_critical_alert', risk_assessment)
            
            # Phase 3: Economic analysis
            logger.info("💰 Phase 3: Economic impact analysis") 
            self.tts.speak("Phase three: Economic impact analysis")
            economic_analysis = self._comprehensive_economic_analysis(data_integration, risk_assessment)
            
            # Phase 4: Decision matrix
            logger.info("🎯 Phase 4: Decision matrix calculation")
            self.tts.speak("Phase four: Decision matrix calculation")
            decision_matrix = self._calculate_decision_matrix(data_integration, risk_assessment, economic_analysis)
            
            # Phase 5: Timing analysis
            logger.info("⏰ Phase 5: Optimal timing analysis")
            timing_analysis = self._analyze_optimal_execution_timing(forecast_result, decision_matrix)
            
            # Phase 6: Final decision
            logger.info("🧠 Phase 6: Final decision synthesis")
            self.tts.speak("Synthesizing final decision")
            final_decision = self._synthesize_final_decision(decision_matrix, timing_analysis, economic_analysis)
            
            # Announce decision immediately
            self.tts.announce_decision(final_decision)
            self._trigger_callbacks('on_decision_made', final_decision)
            
            # Phase 7: Execute if required
            execution_result = None
            if final_decision['execute_cleaning']:
                logger.info("🚿 Phase 7: Executing cleaning action")
                self.tts.speak("Initiating cleaning sequence", alert_level=AlertLevel.WARNING)
                execution_result = await self._execute_cleaning_action_async(final_decision, timing_analysis)
                
                # Announce execution result
                self.tts.announce_execution_result(execution_result)
                self._trigger_callbacks('on_execution_complete', execution_result)
            else:
                logger.info("⏸️ Phase 7: No cleaning action required")
            
            # Phase 8: Generate report
            logger.info("📋 Phase 8: Generating comprehensive report")
            comprehensive_report = self._generate_comprehensive_report(
                data_integration, risk_assessment, economic_analysis,
                decision_matrix, timing_analysis, final_decision, execution_result
            )
            
            # Update performance metrics
            self._update_performance_metrics(final_decision, execution_result)
            
            processing_time = (time.time() - start_time) * 1000
            
            result = {
                'agent_name': self.agent_name,
                'agent_version': self.version,
                'timestamp': datetime.now().isoformat(),
                'success': True,
                'data_integration_analysis': data_integration,
                'multidimensional_risk_assessment': risk_assessment,
                'comprehensive_economic_analysis': economic_analysis,
                'decision_matrix_calculation': decision_matrix,
                'optimal_timing_analysis': timing_analysis,
                'final_decision_synthesis': final_decision,
                'execution_result': execution_result,
                'comprehensive_decision_report': comprehensive_report,
                'processing_time_ms': round(processing_time, 2),
                'performance_metrics': self.performance_metrics
            }
            
            # Final status announcement
            action = "CLEANING EXECUTED" if final_decision['execute_cleaning'] else "NO ACTION REQUIRED"
            confidence = final_decision['decision_confidence']
            
            self.tts.speak(f"Analysis complete. {action}. Confidence level {confidence:.0f} percent.", 
                          alert_level=AlertLevel.SUCCESS)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Decision Orchestration failed: {e}")
            self.tts.speak(f"System error occurred: {str(e)}", 
                          priority=True, alert_level=AlertLevel.CRITICAL)
            self._trigger_callbacks('on_system_error', {'error': str(e)})
            
            return {
                'agent_name': self.agent_name,
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e),
                'final_decision_synthesis': {
                    'execute_cleaning': False,
                    'decision_confidence': 0,
                    'primary_reasoning': f'System error: {str(e)}'
                }
            }
    
    async def _integrate_and_validate_inputs_async(self, dust_result: Dict[str, Any],
                                                  forecast_result: Dict[str, Any],
                                                  image_result: Dict[str, Any]) -> Dict[str, Any]:
        """Async version of data integration with enhanced validation"""
        
        # Simulate async data processing
        await asyncio.sleep(0.1)
        
        # Extract environmental data
        environmental_data = {
            'risk_level': dust_result.get('advanced_risk_assessment', {}).get('risk_level', 'moderate'),
            'risk_score': dust_result.get('advanced_risk_assessment', {}).get('overall_risk_score', 50),
            'predicted_power_loss': dust_result.get('power_impact_prediction', {}).get('estimated_power_loss_percent', 10),
            'urgency_level': dust_result.get('cleaning_urgency_analysis', {}).get('urgency_level', 'moderate'),
            'environmental_confidence': dust_result.get('sensor_reliability', {}).get('overall_reliability', 80)
        }
        
        # Enhanced data validation with voice feedback for critical issues
        if environmental_data['environmental_confidence'] < 50:
            self.tts.speak("Warning: Low sensor confidence detected", alert_level=AlertLevel.WARNING)
        
        # Extract forecast data
        forecast_data = {
            'using_real_quartz': forecast_result.get('quartz_integration_status', {}).get('real_quartz_used', False),
            'daily_loss_kwh': forecast_result.get('dust_corrected_forecast', {}).get('daily_totals', {}).get('total_loss_kwh', 0),
            'cleaning_cost_effective': forecast_result.get('cleaning_impact_analysis', {}).get('recommendations', {}).get('urgency_justified', False),
            'optimal_scenario': forecast_result.get('cleaning_impact_analysis', {}).get('recommendations', {}).get('optimal_scenario', 'standard_cleaning'),
            'forecast_confidence': forecast_result.get('forecast_validation', {}).get('model_validation_metrics', {}).get('validation_confidence', 75)
        }
        
        # Extract visual data with enhanced NPU performance tracking
        visual_data = {
            'dust_level': image_result.get('visual_analysis_results', {}).get('dust_classification', {}).get('primary_level', 'moderate'),
            'visual_confidence': image_result.get('confidence_and_uncertainty', {}).get('overall_confidence', 75),
            'power_impact_visual': image_result.get('power_correlation_analysis', {}).get('estimated_power_impact_percent', 10),
            'image_quality': image_result.get('image_quality_assessment', {}).get('overall_quality_score', 75),
            'npu_accelerated': image_result.get('qualcomm_npu_performance', {}).get('npu_accelerated', False),
            'npu_performance_boost': image_result.get('qualcomm_npu_performance', {}).get('performance_improvement_percent', 0)
        }
        
        # Enhanced consistency validation
        consistency_analysis = self._validate_data_consistency_enhanced(
            environmental_data, forecast_data, visual_data
        )
        
        # Enhanced reliability assessment
        source_reliability = self._assess_source_reliability_enhanced(
            environmental_data, forecast_data, visual_data
        )
        
        return {
            'environmental_data': environmental_data,
            'forecast_data': forecast_data,
            'visual_data': visual_data,
            'data_consistency_analysis': consistency_analysis,
            'source_reliability_assessment': source_reliability,
            'integration_timestamp': datetime.now().isoformat(),
            'data_quality_score': min(100, (consistency_analysis['overall_consistency'] + 
                                           source_reliability['overall_reliability']) / 2),
            'npu_acceleration_status': visual_data['npu_accelerated']
        }
    
    def _validate_data_consistency_enhanced(self, env_data: Dict, forecast_data: Dict, 
                                          visual_data: Dict) -> Dict[str, Any]:
        """Enhanced data consistency validation with ML confidence intervals"""
        
        # Power loss consistency with ML confidence intervals
        env_power_loss = env_data['predicted_power_loss']
        visual_power_loss = visual_data['power_impact_visual']
        forecast_loss_kwh = forecast_data['daily_loss_kwh']
        
        # Convert forecast kWh to percentage with system capacity consideration
        estimated_system_capacity = 30  # kWh - could be configurable
        forecast_power_loss_percent = min(40, (forecast_loss_kwh / estimated_system_capacity) * 100)
        
        power_loss_values = [env_power_loss, visual_power_loss, forecast_power_loss_percent]
        power_loss_std = np.std(power_loss_values)
        power_loss_confidence_interval = [
            np.mean(power_loss_values) - 1.96 * power_loss_std,
            np.mean(power_loss_values) + 1.96 * power_loss_std
        ]
        
        # Enhanced consistency scoring with weighted factors
        power_loss_consistency = max(0, 100 - power_loss_std * 3)
        
        # Risk level consistency with numerical mapping
        risk_levels = {'clean': 1, 'low': 2, 'light': 2, 'moderate': 3, 'high': 4, 'heavy': 4, 'critical': 5}
        env_risk_num = risk_levels.get(env_data['risk_level'], 3)
        visual_risk_num = risk_levels.get(visual_data['dust_level'], 3)
        urgency_risk_num = risk_levels.get(env_data['urgency_level'], 3)
        
        risk_values = [env_risk_num, visual_risk_num, urgency_risk_num]
        risk_consistency = max(0, 100 - np.std(risk_values) * 25)
        
        # Confidence consistency with outlier detection
        confidences = [env_data['environmental_confidence'], 
                      forecast_data['forecast_confidence'], 
                      visual_data['visual_confidence']]
        confidence_consistency = max(0, 100 - np.std(confidences) / 2)
        
        # Overall consistency with weighted average
        weights = [0.4, 0.35, 0.25]  # Power loss most important
        overall_consistency = np.average(
            [power_loss_consistency, risk_consistency, confidence_consistency], 
            weights=weights
        )
        
        return {
            'power_loss_consistency': round(power_loss_consistency, 1),
            'power_loss_confidence_interval': [round(x, 1) for x in power_loss_confidence_interval],
            'risk_level_consistency': round(risk_consistency, 1),
            'confidence_consistency': round(confidence_consistency, 1),
            'overall_consistency': round(overall_consistency, 1),
            'consistency_grade': self._grade_consistency(overall_consistency),
            'inconsistency_flags': self._identify_inconsistency_flags_enhanced(power_loss_values, risk_values),
            'outlier_detection': self._detect_outliers(power_loss_values, confidences)
        }
    
    def _grade_consistency(self, score: float) -> str:
        """Grade consistency score"""
        if score >= 90: return 'excellent'
        elif score >= 80: return 'very_good'
        elif score >= 70: return 'good'
        elif score >= 60: return 'fair'
        elif score >= 50: return 'poor'
        else: return 'very_poor'
    
    def _detect_outliers(self, power_values: List[float], confidence_values: List[float]) -> Dict[str, List]:
        """Detect outliers using IQR method"""
        outliers = {'power_loss': [], 'confidence': []}
        
        for name, values in [('power_loss', power_values), ('confidence', confidence_values)]:
            if len(values) >= 3:
                q1, q3 = np.percentile(values, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers[name] = [v for v in values if v < lower_bound or v > upper_bound]
        
        return outliers
    
    def _assess_source_reliability_enhanced(self, env_data: Dict, forecast_data: Dict, 
                                          visual_data: Dict) -> Dict[str, float]:
        """Enhanced source reliability assessment"""
        
        # Environmental reliability with sensor health factors
        env_base_reliability = env_data['environmental_confidence']
        env_risk_factor = 1.0 if env_data['risk_score'] < 100 else 0.9
        env_reliability = min(100, env_base_reliability * env_risk_factor)
        
        # Forecast reliability with weather model validation
        forecast_base_reliability = forecast_data['forecast_confidence']
        quartz_boost = 1.1 if forecast_data['using_real_quartz'] else 1.0
        forecast_reliability = min(100, forecast_base_reliability * quartz_boost)
        
        # Visual reliability with NPU acceleration consideration
        visual_base_reliability = visual_data['visual_confidence']
        image_quality_factor = visual_data['image_quality'] / 100
        npu_boost = 1.1 if visual_data['npu_accelerated'] else 1.0
        visual_reliability = min(100, visual_base_reliability * image_quality_factor * npu_boost)
        
        # Weighted overall reliability
        reliability_weights = {'environmental': 0.3, 'forecast': 0.35, 'visual': 0.35}
        overall_reliability = (
            env_reliability * reliability_weights['environmental'] +
            forecast_reliability * reliability_weights['forecast'] +
            visual_reliability * reliability_weights['visual']
        )
        
        return {
            'environmental_reliability': round(env_reliability, 1),
            'forecast_reliability': round(forecast_reliability, 1),
            'visual_reliability': round(visual_reliability, 1),
            'overall_reliability': round(overall_reliability, 1),
            'reliability_grade': self._grade_consistency(overall_reliability),
            'critical_reliability_threshold_met': overall_reliability >= 70
        }
    
    def _identify_inconsistency_flags_enhanced(self, power_losses: List[float], 
                                             risk_levels: List[int]) -> List[str]:
        """Enhanced inconsistency flag identification"""
        flags = []
        
        if np.std(power_losses) > 10:
            flags.append('high_power_loss_variance')
        
        if np.std(risk_levels) > 1:
            flags.append('inconsistent_risk_assessment')
        
        if max(power_losses) - min(power_losses) > 20:
            flags.append('extreme_power_loss_disagreement')
        
        if any(pl < 0 for pl in power_losses):
            flags.append('negative_power_loss_detected')
        
        if any(pl > 50 for pl in power_losses):
            flags.append('unrealistic_power_loss_detected')
        
        return flags
    
    async def _execute_cleaning_action_async(self, final_decision: Dict[str, Any],
                                           timing_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced async cleaning execution with voice progress updates
        """
        execution_start = time.time()
        
        try:
            # Pre-execution safety checks with voice updates
            self.tts.speak("Performing pre-execution safety checks", alert_level=AlertLevel.INFO)
            safety_checks = await self._perform_pre_execution_safety_checks_async()
            
            if not safety_checks['all_checks_passed']:
                failure_msg = "Safety checks failed. Cleaning aborted."
                self.tts.speak(failure_msg, priority=True, alert_level=AlertLevel.CRITICAL)
                return {
                    'execution_attempted': True,
                    'execution_successful': False,
                    'failure_reason': 'safety_checks_failed',
                    'safety_check_results': safety_checks,
                    'execution_time_ms': round((time.time() - execution_start) * 1000, 2)
                }
            
            self.tts.speak("Safety checks passed. Beginning cleaning sequence", 
                          alert_level=AlertLevel.SUCCESS)
            
            # Execute cleaning sequence with progress updates
            cleaning_result = await self._execute_cleaning_sequence_async(timing_analysis)
            
            # Post-execution validation
            self.tts.speak("Cleaning complete. Performing validation checks", 
                          alert_level=AlertLevel.INFO)
            post_validation = await self._perform_post_execution_validation_async()
            
            execution_time = (time.time() - execution_start) * 1000
            
            # Update performance metrics
            if cleaning_result['success']:
                self.performance_metrics['successful_executions'] += 1
            else:
                self.performance_metrics['failed_executions'] += 1
            
            return {
                'execution_attempted': True,
                'execution_successful': cleaning_result['success'],
                'pre_execution_safety_checks': safety_checks,
                'cleaning_sequence_result': cleaning_result,
                'post_execution_validation': post_validation,
                'execution_time_ms': round(execution_time, 2),
                'water_usage_planned_liters': cleaning_result['water_usage_planned'],
                'water_usage_actual_liters': cleaning_result['water_usage_actual'],
                'cleaning_duration_seconds': cleaning_result['cleaning_duration'],
                'execution_timestamp': datetime.now().isoformat(),
                'environmental_impact': self._calculate_environmental_impact(cleaning_result)
            }
            
        except Exception as e:
            execution_time = (time.time() - execution_start) * 1000
            error_msg = f"Execution error: {str(e)}"
            self.tts.speak(error_msg, priority=True, alert_level=AlertLevel.CRITICAL)
            self.performance_metrics['failed_executions'] += 1
            
            return {
                'execution_attempted': True,
                'execution_successful': False,
                'failure_reason': 'execution_error',
                'error_details': str(e),
                'execution_time_ms': round(execution_time, 2)
            }
    
    async def _perform_pre_execution_safety_checks_async(self) -> Dict[str, Any]:
        """Enhanced async pre-execution safety checks"""
        
        # Simulate async safety check operations
        await asyncio.sleep(0.5)
        
        safety_checks = {
            'water_pressure_adequate': True,
            'electrical_isolation_confirmed': True,
            'weather_conditions_safe': np.random.choice([True, False], p=[0.9, 0.1]),
            'personnel_safety_cleared': True,
            'equipment_status_operational': True,
            'emergency_stop_functional': True,
            'water_quality_acceptable': True,
            'wind_speed_acceptable': np.random.choice([True, False], p=[0.85, 0.15]),
            'temperature_in_range': True,
            'no_electrical_storms': True
        }
        
        all_passed = all(safety_checks.values())
        failed_checks = [check for check, passed in safety_checks.items() if not passed]
        
        # Voice announcement for failed checks
        if failed_checks:
            failed_list = ', '.join(failed_checks)
            self.tts.speak(f"Safety check failures detected: {failed_list}", 
                          priority=True, alert_level=AlertLevel.CRITICAL)
        
        return {
            'individual_checks': safety_checks,
            'all_checks_passed': all_passed,
            'failed_checks': failed_checks,
            'safety_override_available': not all_passed and len(failed_checks) <= 2,
            'check_timestamp': datetime.now().isoformat(),
            'emergency_contact_notified': len(failed_checks) > 2
        }
    
    async def _execute_cleaning_sequence_async(self, timing_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced async cleaning sequence execution"""
        
        sequence_start = time.time()
        
        # Enhanced planned parameters based on timing analysis
        base_water_usage = 12.5
        timing_factor = timing_analysis.get('recommended_execution_timing', {}).get('suitability_score', 70) / 100
        planned_water_usage = base_water_usage * timing_factor
        planned_duration = 45 * timing_factor
        
        if self.spray_system['simulation_mode']:
            # Enhanced simulation with real-time progress updates
            self.tts.speak("Simulation mode: Beginning spray sequence")
            
            # Simulate multi-stage cleaning process
            stages = ["pre-rinse", "soap application", "scrubbing", "final rinse"]
            total_stages = len(stages)
            
            for i, stage in enumerate(stages):
                progress = (i + 1) / total_stages * 100
                self.tts.speak(f"Stage {i+1}: {stage.replace('_', ' ')}")
                await asyncio.sleep(0.2)  # Simulate stage duration
                logger.info(f"Cleaning progress: {progress:.0f}% - {stage}")
            
            # Simulate realistic variations with weather factors
            weather_factor = np.random.uniform(0.9, 1.1)
            actual_water_usage = planned_water_usage * weather_factor + np.random.normal(0, 1.0)
            actual_duration = planned_duration * weather_factor + np.random.normal(0, 5)
            
            # Higher success probability with enhanced simulation
            success_probability = 0.97
            execution_successful = np.random.random() < success_probability
            
        else:
            # Real hardware execution with enhanced monitoring
            self.tts.speak("Hardware mode: Activating spray system")
            
            try:
                # Real GPIO control implementation would go here
                # Enhanced hardware control with PWM for water pressure
                # import RPi.GPIO as GPIO
                # GPIO.output(self.spray_system['gpio_pin'], GPIO.HIGH)
                
                # Monitor execution in real-time
                for stage_time in [10, 15, 12, 8]:  # Different stage durations
                    await asyncio.sleep(stage_time / 10)  # Scaled for demo
                    logger.info(f"Hardware execution progress...")
                
                # GPIO.output(self.spray_system['gpio_pin'], GPIO.LOW)
                
                actual_water_usage = planned_water_usage
                actual_duration = planned_duration
                execution_successful = True
                
            except Exception as e:
                logger.error(f"Hardware execution failed: {e}")
                self.tts.speak("Hardware execution failed", alert_level=AlertLevel.CRITICAL)
                actual_water_usage = 0
                actual_duration = 0
                execution_successful = False
        
        sequence_time = (time.time() - sequence_start) * 1000
        
        return {
            'success': execution_successful,
            'water_usage_planned': round(planned_water_usage, 1),
            'water_usage_actual': round(max(0, actual_water_usage), 1),
            'cleaning_duration': round(max(0, actual_duration), 1),
            'sequence_execution_time_ms': round(sequence_time, 2),
            'execution_mode': 'simulation' if self.spray_system['simulation_mode'] else 'hardware',
            'spray_pattern': 'adaptive_coverage_sweep',
            'water_pressure_psi': round(35 * timing_factor, 1),
            'nozzle_configuration': 'multi_jet_array_optimized',
            'stages_completed': 4 if execution_successful else np.random.randint(1, 4),
            'efficiency_score': round(85 + np.random.uniform(-10, 15), 1)
        }
    
    async def _perform_post_execution_validation_async(self) -> Dict[str, Any]:
        """Enhanced async post-execution validation"""
        
        # Simulate validation time
        await asyncio.sleep(0.3)
        
        validation_checks = {
            'water_system_shutdown_confirmed': True,
            'no_water_leaks_detected': True,
            'equipment_status_normal': True,
            'cleaning_coverage_adequate': np.random.choice([True, False], p=[0.9, 0.1]),
            'no_damage_observed': True,
            'pressure_system_stable': True,
            'electrical_systems_safe': True,
            'environmental_compliance_met': True
        }
        
        all_validation_passed = all(validation_checks.values())
        failed_validations = [check for check, passed in validation_checks.items() if not passed]
        
        # Enhanced effectiveness estimation with ML model simulation
        base_effectiveness = 0.85 if all_validation_passed else 0.65
        weather_adjustment = np.random.uniform(-0.1, 0.1)
        equipment_adjustment = np.random.uniform(-0.05, 0.15)
        effectiveness_estimate = np.clip(base_effectiveness + weather_adjustment + equipment_adjustment, 0.4, 0.98)
        
        # Voice announcement for validation results
        if all_validation_passed:
            self.tts.speak("All validation checks passed successfully", alert_level=AlertLevel.SUCCESS)
        else:
            failure_list = ', '.join(failed_validations)
            self.tts.speak(f"Validation failures: {failure_list}", alert_level=AlertLevel.WARNING)
        
        return {
            'validation_checks': validation_checks,
            'all_validations_passed': all_validation_passed,
            'failed_validations': failed_validations,
            'estimated_cleaning_effectiveness_percent': round(effectiveness_estimate * 100, 1),
            'recommended_follow_up': self._determine_follow_up_action(all_validation_passed, effectiveness_estimate),
            'validation_timestamp': datetime.now().isoformat(),
            'quality_score': round((len([v for v in validation_checks.values() if v]) / len(validation_checks)) * 100, 1),
            'next_inspection_due': (datetime.now() + timedelta(hours=24)).isoformat()
        }
    
    def _determine_follow_up_action(self, validation_passed: bool, effectiveness: float) -> str:
        """Determine appropriate follow-up action"""
        if validation_passed and effectiveness > 0.8:
            return 'monitor_performance_24h'
        elif validation_passed and effectiveness > 0.6:
            return 'schedule_inspection_12h'
        elif not validation_passed:
            return 'immediate_manual_inspection'
        else:
            return 'consider_repeat_cleaning'
    
    def _calculate_environmental_impact(self, cleaning_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate environmental impact of cleaning operation"""
        
        water_used = cleaning_result.get('water_usage_actual', 0)
        duration = cleaning_result.get('cleaning_duration', 0)
        
        # Water footprint calculation
        water_footprint_score = min(100, water_used * 2)  # Score out of 100
        
        # Energy impact (pump operation)
        estimated_energy_kwh = (duration / 3600) * 2.5  # 2.5 kW pump
        energy_footprint = estimated_energy_kwh * 0.5  # CO2 kg per kWh
        
        # Efficiency vs impact ratio
        efficiency_score = cleaning_result.get('efficiency_score', 80)
        sustainability_ratio = efficiency_score / max(water_footprint_score, 1)
        
        return {
            'water_usage_liters': water_used,
            'estimated_energy_consumption_kwh': round(estimated_energy_kwh, 3),
            'carbon_footprint_kg_co2': round(energy_footprint, 3),
            'water_footprint_score': round(water_footprint_score, 1),
            'sustainability_ratio': round(sustainability_ratio, 2),
            'environmental_grade': self._grade_environmental_impact(sustainability_ratio)
        }
    
    def _grade_environmental_impact(self, sustainability_ratio: float) -> str:
        """Grade environmental impact based on sustainability ratio"""
        if sustainability_ratio >= 2.0: return 'excellent'
        elif sustainability_ratio >= 1.5: return 'good'
        elif sustainability_ratio >= 1.0: return 'fair'
        elif sustainability_ratio >= 0.5: return 'poor'
        else: return 'very_poor'
    
    def _update_performance_metrics(self, final_decision: Dict[str, Any], 
                                  execution_result: Optional[Dict[str, Any]]):
        """Update system performance metrics"""
        self.performance_metrics['total_decisions'] += 1
        
        # Update average confidence
        current_confidence = final_decision.get('decision_confidence', 0)
        total_decisions = self.performance_metrics['total_decisions']
        current_avg = self.performance_metrics['average_confidence']
        
        new_avg = ((current_avg * (total_decisions - 1)) + current_confidence) / total_decisions
        self.performance_metrics['average_confidence'] = round(new_avg, 1)
        
        # Calculate uptime
        uptime_delta = datetime.now() - self.performance_metrics['uptime_start']
        self.performance_metrics['uptime_hours'] = round(uptime_delta.total_seconds() / 3600, 2)
        
        # Calculate success rate
        total_executions = (self.performance_metrics['successful_executions'] + 
                          self.performance_metrics['failed_executions'])
        if total_executions > 0:
            success_rate = (self.performance_metrics['successful_executions'] / total_executions) * 100
            self.performance_metrics['execution_success_rate'] = round(success_rate, 1)
    
    # Include all the original methods with TTS enhancements
    def _perform_multidimensional_risk_assessment(self, data_integration: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced risk assessment with voice alerts for critical risks"""
        
        env_data = data_integration['environmental_data']
        forecast_data = data_integration['forecast_data']
        visual_data = data_integration['visual_data']
        
        # Technical risks
        technical_risks = {
            'power_generation_loss': self._assess_power_generation_risk(env_data, forecast_data, visual_data),
            'equipment_degradation': self._assess_equipment_degradation_risk(env_data, visual_data),
            'system_reliability': self._assess_system_reliability_risk(data_integration),
            'maintenance_complexity': self._assess_maintenance_complexity_risk(visual_data)
        }
        
        # Economic risks
        economic_risks = {
            'revenue_loss': self._assess_revenue_loss_risk(forecast_data),
            'cleaning_cost_escalation': self._assess_cost_escalation_risk(env_data),
            'opportunity_cost': self._assess_opportunity_cost_risk(forecast_data, visual_data),
            'long_term_efficiency': self._assess_long_term_efficiency_risk(env_data, visual_data)
        }
        
        # Operational risks
        operational_risks = {
            'weather_dependency': self._assess_weather_dependency_risk(forecast_data),
            'resource_availability': self._assess_resource_availability_risk(),
            'timing_criticality': self._assess_timing_criticality_risk(env_data, visual_data),
            'execution_complexity': self._assess_execution_complexity_risk(visual_data)
        }
        
        # Environmental risks
        environmental_risks = {
            'dust_accumulation_rate': self._assess_dust_accumulation_risk(env_data),
            'weather_impact': self._assess_weather_impact_risk(forecast_data),
            'seasonal_factors': self._assess_seasonal_risk(),
            'air_quality_trends': self._assess_air_quality_risk(env_data)
        }
        
        # Calculate risk scores
        risk_categories = {
            'technical_risk_score': np.mean(list(technical_risks.values())),
            'economic_risk_score': np.mean(list(economic_risks.values())),
            'operational_risk_score': np.mean(list(operational_risks.values())),
            'environmental_risk_score': np.mean(list(environmental_risks.values()))
        }
        
        # Overall risk calculation
        category_weights = {'technical': 0.3, 'economic': 0.3, 'operational': 0.2, 'environmental': 0.2}
        overall_risk_score = sum(
            risk_categories[f'{category}_risk_score'] * weight 
            for category, weight in category_weights.items()
        )
        
        # Risk classification with voice alerts
        if overall_risk_score >= 80:
            risk_level = 'critical'
            risk_action = 'immediate_action_required'
            self.tts.speak("Critical risk level detected", priority=True, alert_level=AlertLevel.CRITICAL)
        elif overall_risk_score >= 65:
            risk_level = 'high'
            risk_action = 'urgent_action_recommended'
            self.tts.speak("High risk conditions identified", alert_level=AlertLevel.WARNING)
        elif overall_risk_score >= 45:
            risk_level = 'moderate'
            risk_action = 'planned_action_advisable'
        elif overall_risk_score >= 25:
            risk_level = 'low'
            risk_action = 'monitoring_sufficient'
        else:
            risk_level = 'minimal'
            risk_action = 'no_action_needed'
        
        return {
            'technical_risks': technical_risks,
            'economic_risks': economic_risks,
            'operational_risks': operational_risks,
            'environmental_risks': environmental_risks,
            'risk_category_scores': {k: round(v, 1) for k, v in risk_categories.items()},
            'overall_risk_assessment': {
                'overall_risk_score': round(overall_risk_score, 1),
                'risk_level': risk_level,
                'recommended_action': risk_action,
                'risk_tolerance_exceeded': overall_risk_score > self.decision_thresholds['maximum_acceptable_risk']
            },
            'risk_mitigation_priorities': self._identify_risk_mitigation_priorities(
                technical_risks, economic_risks, operational_risks, environmental_risks
            )
        }
    
    # Include risk assessment helper methods (same as original)
    def _assess_power_generation_risk(self, env_data: Dict, forecast_data: Dict, visual_data: Dict) -> float:
        power_losses = [
            env_data['predicted_power_loss'],
            visual_data['power_impact_visual'],
            min(40, (forecast_data['daily_loss_kwh'] / 30) * 100)
        ]
        avg_power_loss = np.mean(power_losses)
        return min(100, avg_power_loss * 2.5)
    
    def _assess_equipment_degradation_risk(self, env_data: Dict, visual_data: Dict) -> float:
        dust_severity = {'clean': 10, 'light': 30, 'moderate': 60, 'heavy': 90}
        visual_severity = dust_severity.get(visual_data['dust_level'], 50)
        env_risk_factor = env_data['risk_score']
        return (visual_severity + env_risk_factor) / 2
    
    def _assess_system_reliability_risk(self, data_integration: Dict) -> float:
        reliability_score = data_integration['source_reliability_assessment']['overall_reliability']
        consistency_score = data_integration['data_consistency_analysis']['overall_consistency']
        return 100 - ((reliability_score + consistency_score) / 2)
    
    def _assess_maintenance_complexity_risk(self, visual_data: Dict) -> float:
        image_quality = visual_data['image_quality']
        dust_level_complexity = {'clean': 10, 'light': 25, 'moderate': 50, 'heavy': 80}
        complexity = dust_level_complexity.get(visual_data['dust_level'], 50)
        quality_factor = max(0, 100 - image_quality) / 2
        return complexity + quality_factor
    
    def _assess_revenue_loss_risk(self, forecast_data: Dict) -> float:
        daily_loss_kwh = forecast_data['daily_loss_kwh']
        daily_loss_usd = daily_loss_kwh * self.economic_params['electricity_rate_per_kwh']
        return min(100, daily_loss_usd * 10)
    
    def _assess_cost_escalation_risk(self, env_data: Dict) -> float:
        urgency_escalation = {'low': 0, 'moderate': 20, 'high': 50, 'critical_immediate': 80}
        return urgency_escalation.get(env_data['urgency_level'], 20)
    
    def _assess_opportunity_cost_risk(self, forecast_data: Dict, visual_data: Dict) -> float:
        if not forecast_data['cleaning_cost_effective']:
            return 70
        visual_confidence = visual_data['visual_confidence']
        return max(0, 80 - visual_confidence)
    
    def _assess_long_term_efficiency_risk(self, env_data: Dict, visual_data: Dict) -> float:
        dust_accumulation_factor = {'clean': 5, 'light': 20, 'moderate': 50, 'heavy': 85}
        env_risk_factor = env_data['risk_score'] / 2
        visual_risk_factor = dust_accumulation_factor.get(visual_data['dust_level'], 40)
        return (env_risk_factor + visual_risk_factor) / 2
    
    def _assess_weather_dependency_risk(self, forecast_data: Dict) -> float:
        forecast_confidence = forecast_data['forecast_confidence']
        using_real_quartz = forecast_data['using_real_quartz']
        base_risk = 100 - forecast_confidence
        if not using_real_quartz:
            base_risk += 20
        return min(100, base_risk)
    
    def _assess_resource_availability_risk(self) -> float:
        water_availability = 95
        equipment_availability = 90
        labor_availability = 85
        avg_availability = (water_availability + equipment_availability + labor_availability) / 3
        return 100 - avg_availability
    
    def _assess_timing_criticality_risk(self, env_data: Dict, visual_data: Dict) -> float:
        urgency_scores = {'low': 10, 'moderate': 30, 'high': 70, 'critical_immediate': 95}
        urgency_risk = urgency_scores.get(env_data['urgency_level'], 30)
        dust_timing_risk = {'clean': 5, 'light': 15, 'moderate': 40, 'heavy': 80}
        dust_risk = dust_timing_risk.get(visual_data['dust_level'], 40)
        return max(urgency_risk, dust_risk)
    
    def _assess_execution_complexity_risk(self, visual_data: Dict) -> float:
        image_quality = visual_data['image_quality']
        npu_accelerated = visual_data['npu_accelerated']
        complexity_risk = max(0, 100 - image_quality) / 2
        if npu_accelerated:
            complexity_risk *= 0.8
        return complexity_risk
    
    def _assess_dust_accumulation_risk(self, env_data: Dict) -> float:
        return env_data['risk_score']
    
    def _assess_weather_impact_risk(self, forecast_data: Dict) -> float:
        return max(0, 80 - forecast_data['forecast_confidence'])
    
    def _assess_seasonal_risk(self) -> float:
        current_month = datetime.now().month
        if current_month in [3, 4, 5, 10, 11]:
            return 70
        elif current_month in [6, 7, 8, 9]:
            return 30
        else:
            return 50
    
    def _assess_air_quality_risk(self, env_data: Dict) -> float:
        return env_data['risk_score']
    
    def _identify_risk_mitigation_priorities(self, technical: Dict, economic: Dict, 
                                           operational: Dict, environmental: Dict) -> List[Dict]:
        all_risks = []
        for category, risks in [('technical', technical), ('economic', economic), 
                               ('operational', operational), ('environmental', environmental)]:
            for risk_name, risk_score in risks.items():
                all_risks.append({
                    'category': category,
                    'risk_name': risk_name,
                    'risk_score': risk_score,
                    'priority': 'critical' if risk_score > 80 else 'high' if risk_score > 60 else 'medium'
                })
        
        all_risks.sort(key=lambda x: x['risk_score'], reverse=True)
        return all_risks[:5]
    
    # Include simplified versions of other original methods for brevity
    def _comprehensive_economic_analysis(self, data_integration: Dict[str, Any],
                                       risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Simplified economic analysis for demo"""
        forecast_data = data_integration['forecast_data']
        
        daily_loss_kwh = forecast_data['daily_loss_kwh']
        electricity_rate = self.economic_params['electricity_rate_per_kwh']
        
        revenue_impact = {
            'daily_loss_kwh': daily_loss_kwh,
            'daily_revenue_loss_usd': round(daily_loss_kwh * electricity_rate, 2),
            'weekly_revenue_loss_usd': round(daily_loss_kwh * electricity_rate * 7, 2),
            'monthly_revenue_loss_usd': round(daily_loss_kwh * electricity_rate * 30, 2),
            'annual_revenue_loss_usd': round(daily_loss_kwh * electricity_rate * 365, 2)
        }
        
        cleaning_costs = {
            'total_cleaning_cost_usd': 25.50,
            'water_cost_usd': 2.50,
            'labor_cost_usd': 15.00,
            'equipment_cost_usd': 8.00
        }
        
        roi_analysis = {
            'economic_viability': revenue_impact['daily_revenue_loss_usd'] * 7 > cleaning_costs['total_cleaning_cost_usd'],
            'payback_period_days': cleaning_costs['total_cleaning_cost_usd'] / max(revenue_impact['daily_revenue_loss_usd'], 0.01)
        }
        
        return {
            'revenue_impact_analysis': revenue_impact,
            'comprehensive_cleaning_costs': cleaning_costs,
            'roi_scenario_analysis': roi_analysis
        }
    
    def _calculate_decision_matrix(self, data_integration: Dict[str, Any],
                                 risk_assessment: Dict[str, Any],
                                 economic_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Simplified decision matrix calculation"""
        
        visual_data = data_integration['visual_data']
        overall_confidence = data_integration['source_reliability_assessment']['overall_reliability']
        overall_risk = risk_assessment['overall_risk_assessment']['overall_risk_score']
        
        # Calculate decision factors
        decision_factors = {
            'dust_severity': {
                'score': {'clean': 15, 'light': 40, 'moderate': 70, 'heavy': 95}.get(visual_data['dust_level'], 50),
                'weight': self.decision_framework['dust_severity_weight']
            },
            'economic_impact': {
                'score': 80 if economic_analysis['roi_scenario_analysis']['economic_viability'] else 40,
                'weight': self.decision_framework['economic_impact_weight']
            },
            'risk_assessment': {
                'score': overall_risk,
                'weight': self.decision_framework['risk_assessment_weight']
            },
            'confidence_level': {
                'score': overall_confidence,
                'weight': self.decision_framework['confidence_weight']
            },
            'timing_optimization': {
                'score': 70,  # Simplified
                'weight': self.decision_framework['timing_optimization_weight']
            }
        }
        
        # Calculate weighted total score
        total_score = sum(
            factor['score'] * factor['weight'] 
            for factor in decision_factors.values()
        )
        
        # Classify decision
        if total_score >= self.decision_thresholds['critical_action_score']:
            decision_class = 'critical_action_required'
        elif total_score >= self.decision_thresholds['recommended_action_score']:
            decision_class = 'action_recommended'
        elif total_score >= self.decision_thresholds['optional_action_score']:
            decision_class = 'action_optional'
        else:
            decision_class = 'no_action_needed'
        
        return {
            'decision_factors': {k: {**v, 'score': round(v['score'], 1)} for k, v in decision_factors.items()},
            'total_decision_score': round(total_score, 1),
            'decision_classification': decision_class,
            'validation_results': {'validation_passed': overall_confidence >= 60}
        }
    
    def _analyze_optimal_execution_timing(self, forecast_result: Dict[str, Any],
                                        decision_matrix: Dict[str, Any]) -> Dict[str, Any]:
        """Simplified timing analysis"""
        decision_class = decision_matrix['decision_classification']
        
        if decision_class == 'critical_action_required':
            timing_urgency = 'immediate'
        elif decision_class == 'action_recommended':
            timing_urgency = 'within_24_hours'
        else:
            timing_urgency = 'within_week'
        
        return {
            'timing_urgency': timing_urgency,
            'recommended_execution_timing': {
                'timing_type': 'immediate_execution' if timing_urgency == 'immediate' else 'scheduled',
                'start_time': datetime.now().isoformat(),
                'suitability_score': 80
            },
            'execution_readiness': {'ready_for_execution': True}
        }
    
    def _synthesize_final_decision(self, decision_matrix: Dict[str, Any],
                                 timing_analysis: Dict[str, Any],
                                 economic_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced final decision synthesis with voice feedback"""
        
        decision_score = decision_matrix['total_decision_score']
        decision_class = decision_matrix['decision_classification']
        economic_viable = economic_analysis['roi_scenario_analysis']['economic_viability']
        
        if decision_score >= self.decision_thresholds['recommended_action_score'] and economic_viable:
            execute_cleaning = True
            decision_confidence = min(95, decision_score)
            primary_reasoning = f"High decision score ({decision_score:.1f}) with economic justification"
        elif decision_score >= self.decision_thresholds['critical_action_score']:
            execute_cleaning = True
            decision_confidence = min(90, decision_score - 5)
            primary_reasoning = f"Critical conditions override economic concerns"
        else:
            execute_cleaning = False
            decision_confidence = max(60, 100 - decision_score)
            primary_reasoning = f"Insufficient justification for cleaning action"
        
        return {
            'execute_cleaning': execute_cleaning,
            'decision_confidence': round(decision_confidence, 1),
            'primary_reasoning': primary_reasoning,
            'decision_classification': decision_class,
            'alternative_recommendations': ['Continue monitoring', 'Schedule maintenance check']
        }
    
    def _generate_comprehensive_report(self, data_integration: Dict[str, Any],
                                     risk_assessment: Dict[str, Any],
                                     economic_analysis: Dict[str, Any],
                                     decision_matrix: Dict[str, Any],
                                     timing_analysis: Dict[str, Any],
                                     final_decision: Dict[str, Any],
                                     execution_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive report with voice summary"""
        
        executive_summary = {
            'decision_summary': f"{'CLEANING EXECUTED' if final_decision['execute_cleaning'] else 'NO ACTION'} with {final_decision['decision_confidence']:.1f}% confidence",
            'primary_reasoning': final_decision['primary_reasoning'],
            'execution_outcome': 'Successfully completed' if execution_result and execution_result.get('execution_successful') else 'No execution or failed',
            'economic_justification': f"Economic viability: {'YES' if economic_analysis['roi_scenario_analysis']['economic_viability'] else 'NO'}",
            'key_recommendation': final_decision.get('alternative_recommendations', ['Continue monitoring'])[0]
        }
        
        # Voice summary of report
        summary_text = f"Report generated. {executive_summary['decision_summary']}. {executive_summary['primary_reasoning']}"
        self.tts.speak(summary_text, alert_level=AlertLevel.INFO)
        
        return {
            'executive_summary': executive_summary,
            'key_performance_indicators': {
                'decision_confidence': final_decision['decision_confidence'],
                'risk_level': risk_assessment['overall_risk_assessment']['risk_level'],
                'economic_impact': economic_analysis['revenue_impact_analysis']['daily_revenue_loss_usd']
            },
            'report_metadata': {
                'generated_timestamp': datetime.now().isoformat(),
                'report_version': '3.0',
                'system_version': self.version
            }
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status with voice announcement"""
        
        uptime_delta = datetime.now() - self.performance_metrics['uptime_start']
        uptime_hours = uptime_delta.total_seconds() / 3600
        
        status = {
            'system_operational': True,
            'tts_enabled': self.tts.config.enabled,
            'spray_system_mode': 'simulation' if self.spray_system['simulation_mode'] else 'hardware',
            'uptime_hours': round(uptime_hours, 2),
            'total_decisions': self.performance_metrics['total_decisions'],
            'average_confidence': self.performance_metrics['average_confidence'],
            'execution_success_rate': self.performance_metrics.get('execution_success_rate', 0)
        }
        
        # Voice status update if requested
        status_message = f"System operational. Uptime {uptime_hours:.1f} hours. {self.performance_metrics['total_decisions']} decisions processed."
        self.tts.speak(status_message, alert_level=AlertLevel.INFO)
        
        return status
    
    def emergency_shutdown(self):
        """Emergency shutdown with voice alert"""
        
        self.tts.speak("Emergency shutdown initiated. All systems stopping immediately.", 
                      priority=True, alert_level=AlertLevel.CRITICAL)
        
        # Stop any ongoing operations
        if hasattr(self, 'current_operation'):
            self.current_operation = None
        
        # Hardware shutdown procedures would go here
        if not self.spray_system['simulation_mode']:
            # GPIO.output(self.spray_system['gpio_pin'], GPIO.LOW)
            pass
        
        logger.critical("Emergency shutdown completed")
        self.tts.speak("Emergency shutdown completed. System is safe.", alert_level=AlertLevel.INFO)


# Enhanced main function with TTS demonstrations
async def main():
    """
    Enhanced main function demonstrating TTS capabilities
    """
    print("=" * 80)
    print("ENHANCED SOLAR PANEL CLEANING DECISION ORCHESTRATION AGENT")
    print("With Text-to-Speech and Advanced Features")
    print("=" * 80)
    print(f"Version: 3.0.0")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 80)
    
    # Configure TTS
    tts_config = TTSConfig(
        engine=TTSEngine.PYTTSX3,
        rate=160,
        volume=0.8,
        enabled=True,
        audio_alerts=True
    )
    
    # Create enhanced decision agent
    agent = EnhancedDecisionOrchestrationAgent(tts_config=tts_config)
    
    # Demonstrate system status
    print("\n🔊 Getting system status...")
    status = agent.get_system_status()
    print(f"System Status: {status}")
    
    # Mock input data for comprehensive testing
    mock_dust_result = {
        'advanced_risk_assessment': {
            'risk_level': 'high',
            'overall_risk_score': 78
        },
        'power_impact_prediction': {
            'estimated_power_loss_percent': 18
        },
        'cleaning_urgency_analysis': {
            'urgency_level': 'high'
        },
        'sensor_reliability': {
            'overall_reliability': 87
        }
    }
    
    mock_forecast_result = {
        'quartz_integration_status': {
            'real_quartz_used': True
        },
        'dust_corrected_forecast': {
            'daily_totals': {
                'total_loss_kwh': 4.2
            }
        },
        'cleaning_impact_analysis': {
            'recommendations': {
                'urgency_justified': True,
                'optimal_scenario': 'immediate_cleaning'
            }
        },
        'forecast_validation': {
            'model_validation_metrics': {
                'validation_confidence': 82
            }
        },
        'weather_optimization_analysis': {
            'optimal_cleaning_windows': [
                {
                    'start_time': '2024-01-15T10:00:00',
                    'duration_hours': 3,
                    'suitability_score': 87
                }
            ],
            'cleaning_favorability': 'excellent',
            'weather_pattern_classification': 'stable_high_pressure'
        }
    }
    
    mock_image_result = {
        'visual_analysis_results': {
            'dust_classification': {
                'primary_level': 'moderate'
            }
        },
        'confidence_and_uncertainty': {
            'overall_confidence': 83
        },
        'power_correlation_analysis': {
            'estimated_power_impact_percent': 15
        },
        'image_quality_assessment': {
            'overall_quality_score': 88
        },
        'qualcomm_npu_performance': {
            'npu_accelerated': True,
            'performance_improvement_percent': 35
        }
    }
    
    # Register some demo callbacks
    def on_decision_callback(decision_data):
        print(f"🔔 Decision callback triggered: {decision_data['execute_cleaning']}")
    
    def on_execution_callback(execution_data):
        print(f"🔔 Execution callback triggered: {execution_data['execution_successful']}")
    
    agent.register_callback('on_decision_made', on_decision_callback)
    agent.register_callback('on_execution_complete', on_execution_callback)
    
    # Execute comprehensive decision analysis with TTS
    print("\n🎯 Starting Enhanced Comprehensive Decision Analysis with Voice Feedback...")
    result = await agent.orchestrate_comprehensive_decision_async(
        mock_dust_result, 
        mock_forecast_result, 
        mock_image_result
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("ENHANCED DECISION ORCHESTRATION SUMMARY")
    print("=" * 60)
    
    if result['success']:
        final_decision = result['final_decision_synthesis']
        
        print(f"🎯 Final Decision: {'EXECUTE CLEANING' if final_decision['execute_cleaning'] else 'NO ACTION'}")
        print(f"📊 Confidence Level: {final_decision['decision_confidence']:.1f}%")
        print(f"💡 Primary Reasoning: {final_decision['primary_reasoning']}")
        print(f"⚖️ Decision Class: {final_decision['decision_classification']}")
        
        # Economic summary
        economic = result['comprehensive_economic_analysis']
        daily_loss = economic['revenue_impact_analysis']['daily_revenue_loss_usd']
        cleaning_cost = economic['comprehensive_cleaning_costs']['total_cleaning_cost_usd']
        roi_viable = economic['roi_scenario_analysis']['economic_viability']
        
        print(f"\n💰 Economic Analysis:")
        print(f"   Daily Revenue Loss: ${daily_loss:.2f}")
        print(f"   Cleaning Cost: ${cleaning_cost:.2f}")
        print(f"   Economic Viability: {'YES' if roi_viable else 'NO'}")
        
        # Risk summary
        risk = result['multidimensional_risk_assessment']
        risk_score = risk['overall_risk_assessment']['overall_risk_score']
        risk_level = risk['overall_risk_assessment']['risk_level']
        
        print(f"\n⚠️ Risk Assessment:")
        print(f"   Overall Risk Score: {risk_score:.1f}/100")
        print(f"   Risk Level: {risk_level.upper()}")
        
        # Enhanced execution results
        if result.get('execution_result'):
            exec_result = result['execution_result']
            success = exec_result['execution_successful']
            water_used = exec_result.get('water_usage_actual_liters', 0)
            env_impact = exec_result.get('environmental_impact', {})
            
            print(f"\n🚿 Execution Results:")
            print(f"   Status: {'SUCCESS' if success else 'FAILED'}")
            if success:
                print(f"   Water Used: {water_used:.1f}L")
                print(f"   Duration: {exec_result.get('cleaning_duration', 0):.1f}s")
                print(f"   Efficiency Score: {exec_result.get('efficiency_score', 0):.1f}%")
                if env_impact:
                    print(f"   Environmental Grade: {env_impact.get('environmental_grade', 'N/A')}")
                    print(f"   Carbon Footprint: {env_impact.get('carbon_footprint_kg_co2', 0):.3f} kg CO₂")
        
        # Performance metrics
        perf_metrics = result.get('performance_metrics', {})
        print(f"\n📈 Performance Metrics:")
        print(f"   Total Decisions: {perf_metrics.get('total_decisions', 0)}")
        print(f"   Average Confidence: {perf_metrics.get('average_confidence', 0):.1f}%")
        print(f"   Success Rate: {perf_metrics.get('execution_success_rate', 0):.1f}%")
        print(f"   System Uptime: {perf_metrics.get('uptime_hours', 0):.2f} hours")
        
        print(f"\n⏱️ Processing Time: {result['processing_time_ms']:.1f}ms")
        
        # TTS and enhancement features
        print(f"\n🔊 Enhanced Features:")
        print(f"   TTS Enabled: {agent.tts.config.enabled}")
        print(f"   TTS Engine: {agent.tts.config.engine.value}")
        print(f"   Audio Alerts: {agent.tts.config.audio_alerts}")
        print(f"   NPU Acceleration: {result['data_integration_analysis'].get('npu_acceleration_status', False)}")
        
    else:
        print(f"❌ Enhanced decision orchestration failed: {result.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 60)
    print("Enhanced Analysis Complete - Voice Feedback Provided")
    print("=" * 60)
    
    # Demonstrate emergency shutdown
    print("\n🚨 Demonstrating Emergency Shutdown...")
    await asyncio.sleep(1)
    agent.emergency_shutdown()


if __name__ == "__main__":
    # Run async main function
    asyncio.run(main())