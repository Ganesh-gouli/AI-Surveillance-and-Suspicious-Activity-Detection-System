/**
 * Tactical Web Audio API Alert Synthesizer for SentinelVision AI.
 * Creates procedural sounds without external audio assets.
 */

class TacticalAudioService {
  private ctx: AudioContext | null = null;
  private isMuted: boolean = false;

  constructor() {
    const saved = localStorage.getItem('sentinel_audio_muted');
    this.isMuted = saved === 'true';
  }

  private getContext(): AudioContext | null {
    if (typeof window === 'undefined') return null;
    if (!this.ctx) {
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      if (AudioCtx) {
        this.ctx = new AudioCtx();
      }
    }
    if (this.ctx && this.ctx.state === 'suspended') {
      this.ctx.resume();
    }
    return this.ctx;
  }

  public setMuted(muted: boolean) {
    this.isMuted = muted;
    localStorage.setItem('sentinel_audio_muted', String(muted));
  }

  public toggleMute(): boolean {
    this.isMuted = !this.isMuted;
    localStorage.setItem('sentinel_audio_muted', String(this.isMuted));
    return this.isMuted;
  }

  public getIsMuted(): boolean {
    return this.isMuted;
  }

  /**
   * Plays a high-priority warning blip (e.g. for medium threats or loitering).
   */
  public playWarningBeep() {
    if (this.isMuted) return;
    const ctx = this.getContext();
    if (!ctx) return;

    try {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = 'triangle';
      osc.frequency.setValueAtTime(880, ctx.currentTime); // A5
      osc.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.18);

      gain.gain.setValueAtTime(0.15, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.18);

      osc.connect(gain);
      gain.connect(ctx.destination);

      osc.start();
      osc.stop(ctx.currentTime + 0.18);
    } catch (e) {
      console.warn("Audio synthesis note:", e);
    }
  }

  /**
   * Plays an urgent multi-frequency dual-pulse alarm for CRITICAL threats (Fighting, Breach).
   */
  public playCriticalAlarm() {
    if (this.isMuted) return;
    const ctx = this.getContext();
    if (!ctx) return;

    try {
      const now = ctx.currentTime;
      // Pulse 1
      const osc1 = ctx.createOscillator();
      const gain1 = ctx.createGain();
      osc1.type = 'sawtooth';
      osc1.frequency.setValueAtTime(960, now);
      osc1.frequency.linearRampToValueAtTime(600, now + 0.25);
      gain1.gain.setValueAtTime(0.25, now);
      gain1.gain.exponentialRampToValueAtTime(0.001, now + 0.25);
      osc1.connect(gain1);
      gain1.connect(ctx.destination);
      osc1.start(now);
      osc1.stop(now + 0.25);

      // Pulse 2
      const osc2 = ctx.createOscillator();
      const gain2 = ctx.createGain();
      osc2.type = 'sawtooth';
      osc2.frequency.setValueAtTime(1100, now + 0.3);
      osc2.frequency.linearRampToValueAtTime(700, now + 0.55);
      gain2.gain.setValueAtTime(0.25, now + 0.3);
      gain2.gain.exponentialRampToValueAtTime(0.001, now + 0.55);
      osc2.connect(gain2);
      gain2.connect(ctx.destination);
      osc2.start(now + 0.3);
      osc2.stop(now + 0.55);
    } catch (e) {
      console.warn("Audio synthesis note:", e);
    }
  }

  /**
   * Tactical Voice Synthesizer Announcement.
   */
  public speakAnnouncement(text: string) {
    if (this.isMuted) return;
    if ('speechSynthesis' in window) {
      try {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.05;
        utterance.pitch = 0.95;
        utterance.volume = 0.8;
        window.speechSynthesis.speak(utterance);
      } catch (e) {
        console.warn("Speech synthesis note:", e);
      }
    }
  }
}

export const audioService = new TacticalAudioService();
