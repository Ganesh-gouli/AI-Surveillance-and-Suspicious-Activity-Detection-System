import React, { useRef, useEffect } from 'react';
import { TrackedPerson, RestrictedZone, DetectedObject } from '../../types';

interface CameraCanvasOverlayProps {
  people: TrackedPerson[];
  objects?: DetectedObject[];
  zones?: RestrictedZone[];
  showBoundingBoxes?: boolean;
  showTrackingIds?: boolean;
  showActivityLabels?: boolean;
  showObjects?: boolean;
  showZones?: boolean;
  showSkeleton?: boolean;
  videoWidth?: number;
  videoHeight?: number;
  highlightThreats?: boolean;
}

const CATEGORY_COLORS: Record<string, { stroke: string; fill: string }> = {
  electronics: { stroke: '#c084fc', fill: 'rgba(192, 132, 252, 0.15)' },
  sustenance: { stroke: '#34d399', fill: 'rgba(52, 211, 153, 0.15)' },
  mobility: { stroke: '#38bdf8', fill: 'rgba(56, 189, 248, 0.15)' },
  luggage: { stroke: '#fbbf24', fill: 'rgba(251, 191, 36, 0.15)' },
  threat: { stroke: '#f87171', fill: 'rgba(248, 113, 113, 0.25)' },
  threat_or_dining: { stroke: '#fb923c', fill: 'rgba(251, 146, 60, 0.18)' },
  furniture: { stroke: '#94a3b8', fill: 'rgba(148, 163, 184, 0.12)' },
  sports: { stroke: '#2dd4bf', fill: 'rgba(45, 212, 191, 0.15)' },
  general_object: { stroke: '#818cf8', fill: 'rgba(129, 140, 248, 0.15)' }
};

// Anatomical bone connections for wireframe rendering (MediaPipe Pose 33 landmarks)
const POSE_CONNECTIONS: [number, number][] = [
  // Face
  [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [3, 7], [6, 8],
  // Shoulders & Torso
  [11, 12], [11, 23], [12, 24], [23, 24],
  // Left Arm
  [11, 13], [13, 15], [15, 17], [15, 19], [15, 21],
  // Right Arm
  [12, 14], [14, 16], [16, 18], [16, 20], [16, 22],
  // Left Leg
  [23, 25], [25, 27], [27, 29], [27, 31], [29, 31],
  // Right Leg
  [24, 26], [26, 28], [28, 30], [28, 32], [30, 32]
];

// Safe benign activities that must NEVER trigger red colors or suspicious flags
const SAFE_ACTIVITIES = new Set([
  'SITTING', 'STANDING', 'SLEEPING', 'USING_LAPTOP', 'USING_PHONE',
  'DRINKING', 'EATING', 'WALKING', 'CARRYING_BAG', 'PLAYING_SPORTS'
]);

interface SmoothedPerson {
  x: number;
  y: number;
  w: number;
  h: number;
  landmarks: { x: number; y: number }[];
}

export const CameraCanvasOverlay: React.FC<CameraCanvasOverlayProps> = ({
  people = [],
  objects = [],
  zones = [],
  showBoundingBoxes = true,
  showTrackingIds = true,
  showActivityLabels = true,
  showObjects = true,
  showZones = true,
  showSkeleton = true,
  videoWidth = 1280,
  videoHeight = 720,
  highlightThreats = true,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const smoothedRef = useRef<Map<number, SmoothedPerson>>(new Map());
  const animFrameRef = useRef<number | null>(null);

  // 60 FPS Smooth Rendering Loop with LERP interpolation
  useEffect(() => {
    let active = true;

    const render = () => {
      if (!active) return;
      const canvas = canvasRef.current;
      if (!canvas) {
        animFrameRef.current = requestAnimationFrame(render);
        return;
      }
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        animFrameRef.current = requestAnimationFrame(render);
        return;
      }

      const scaleX = canvas.width / videoWidth;
      const scaleY = canvas.height / videoHeight;

      // Clear previous frame
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // 1. Draw Restricted Zones (Polygons)
      if (showZones && zones.length > 0) {
        zones.forEach((zone) => {
          if (!zone.is_active || zone.polygon.length < 3) return;

          ctx.save();
          ctx.beginPath();
          const startX = (zone.polygon[0].x / 100) * canvas.width;
          const startY = (zone.polygon[0].y / 100) * canvas.height;
          ctx.moveTo(startX, startY);

          for (let i = 1; i < zone.polygon.length; i++) {
            const px = (zone.polygon[i].x / 100) * canvas.width;
            const py = (zone.polygon[i].y / 100) * canvas.height;
            ctx.lineTo(px, py);
          }
          ctx.closePath();

          const isCritical = zone.severity === 'CRITICAL';
          ctx.fillStyle = isCritical ? 'rgba(239, 68, 68, 0.15)' : 'rgba(245, 158, 11, 0.12)';
          ctx.fill();

          ctx.lineWidth = 1.5;
          ctx.strokeStyle = isCritical ? '#ef4444' : '#f59e0b';
          ctx.setLineDash([6, 4]);
          ctx.stroke();

          ctx.font = 'bold 11px JetBrains Mono, monospace';
          ctx.fillStyle = isCritical ? '#ef4444' : '#f59e0b';
          ctx.fillText(`☡ ${zone.name.toUpperCase()} [${zone.severity}]`, startX + 6, startY + 16);
          ctx.restore();
        });
      }

      // 2. Draw Standalone Scene Objects
      if (showObjects && objects.length > 0) {
        objects.forEach((obj) => {
          const ox = obj.bbox.x * scaleX;
          const oy = obj.bbox.y * scaleY;
          const ow = obj.bbox.width * scaleX;
          const oh = obj.bbox.height * scaleY;

          const colorCfg = CATEGORY_COLORS[obj.category] || CATEGORY_COLORS.general_object;

          ctx.save();
          ctx.lineWidth = 1.5;
          ctx.strokeStyle = colorCfg.stroke;
          ctx.fillStyle = colorCfg.fill;
          ctx.fillRect(ox, oy, ow, oh);
          ctx.strokeRect(ox, oy, ow, oh);

          ctx.font = 'bold 10px JetBrains Mono, monospace';
          const objText = `${obj.label.toUpperCase()} ${obj.confidence.toFixed(0)}%`;
          const tagW = ctx.measureText(objText).width + 8;
          
          ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
          ctx.fillRect(ox, Math.max(0, oy - 16), tagW, 16);
          ctx.strokeStyle = colorCfg.stroke;
          ctx.strokeRect(ox, Math.max(0, oy - 16), tagW, 16);
          ctx.fillStyle = colorCfg.stroke;
          ctx.fillText(objText, ox + 4, Math.max(0, oy - 16) + 12);
          ctx.restore();
        });
      }

      // 3. Draw Tracked People with Smooth Coordinate Interpolation & MediaPipe Skeleton
      if (people.length > 0) {
        const smoothedMap = smoothedRef.current;
        const alpha = 0.35; // LERP smoothing factor (fluid, zero-lag response)

        people.forEach((p) => {
          const targetX = p.bbox.x * scaleX;
          const targetY = p.bbox.y * scaleY;
          const targetW = p.bbox.width * scaleX;
          const targetH = p.bbox.height * scaleY;

          // Target landmarks
          const targetLms = (p.landmarks || []).map((lm) => ({
            x: (lm.pixel_x !== undefined ? lm.pixel_x : lm.x * videoWidth) * scaleX,
            y: (lm.pixel_y !== undefined ? lm.pixel_y : lm.y * videoHeight) * scaleY
          }));

          let current = smoothedMap.get(p.track_id);
          if (!current) {
            current = { x: targetX, y: targetY, w: targetW, h: targetH, landmarks: targetLms };
            smoothedMap.set(p.track_id, current);
          } else {
            // Smoothly interpolate towards new target
            current.x += (targetX - current.x) * alpha;
            current.y += (targetY - current.y) * alpha;
            current.w += (targetW - current.w) * alpha;
            current.h += (targetH - current.h) * alpha;

            if (targetLms.length === current.landmarks.length) {
              for (let i = 0; i < targetLms.length; i++) {
                current.landmarks[i].x += (targetLms[i].x - current.landmarks[i].x) * alpha;
                current.landmarks[i].y += (targetLms[i].y - current.landmarks[i].y) * alpha;
              }
            } else {
              current.landmarks = targetLms;
            }
          }

          const x = current.x;
          const y = current.y;
          const w = current.w;
          const h = current.h;

          // STRICT SAFETY LOGIC:
          // Sitting, standing, sleeping are NEVER suspicious and NEVER colored red!
          // Only actual dangerous activities (fighting, weapons) turn red.
          const upperActivity = p.activity.toUpperCase();
          const isSafeActivity = SAFE_ACTIVITIES.has(upperActivity);

          const hasWeapon = (p.interacting_objects || []).some(
            (o) => o.category === 'threat' || ['knife', 'scissors', 'weapon'].includes(o.label.toLowerCase())
          );
          const isFighting = upperActivity.includes('FIGHT');

          const isThreat = !isSafeActivity && (isFighting || hasWeapon || p.threat_level === 'CRITICAL');

          // Palette selection based on posture / activity
          let mainColor = '#00f0ff'; // Vibrant Cyber Cyan default
          if (isThreat) {
            mainColor = '#ef4444'; // Red ONLY on true dangerous threats
          } else if (upperActivity === 'STANDING') {
            mainColor = '#10b981'; // Calm Emerald for Standing
          } else if (upperActivity === 'SITTING') {
            mainColor = '#00f0ff'; // Neon Cyan for Sitting
          } else if (upperActivity === 'SLEEPING') {
            mainColor = '#a855f7'; // Peaceful Violet for Sleeping
          } else if (upperActivity.includes('LAPTOP') || upperActivity.includes('PHONE')) {
            mainColor = '#38bdf8'; // Sky Blue for tech interactions
          }

          // A. Draw MediaPipe 3D Skeletal Wireframe
          if (showSkeleton && current.landmarks.length >= 25) {
            const lms = current.landmarks;
            ctx.save();
            ctx.lineWidth = 2.0;

            // Draw anatomical bone segments
            POSE_CONNECTIONS.forEach(([i, j]) => {
              if (i < lms.length && j < lms.length) {
                const p1 = lms[i];
                const p2 = lms[j];
                if (p1.x > 0 && p1.y > 0 && p2.x > 0 && p2.y > 0) {
                  ctx.beginPath();
                  ctx.moveTo(p1.x, p1.y);
                  ctx.lineTo(p2.x, p2.y);
                  
                  // Color branches: Legs in sky blue, Torso in emerald, Face/arms in cyan
                  if (i >= 23 || j >= 23) {
                    ctx.strokeStyle = 'rgba(56, 189, 248, 0.75)';
                  } else if (i >= 11 && j >= 11) {
                    ctx.strokeStyle = 'rgba(16, 185, 129, 0.85)';
                  } else {
                    ctx.strokeStyle = 'rgba(0, 240, 255, 0.75)';
                  }
                  ctx.stroke();
                }
              }
            });

            // Draw joint nodes
            lms.forEach((lm, idx) => {
              if (lm.x > 0 && lm.y > 0) {
                // Key anatomical joints (shoulders, elbows, wrists, hips, knees, ankles)
                const isKeyJoint = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28].includes(idx);
                const radius = isKeyJoint ? 3.5 : 2.0;

                ctx.beginPath();
                ctx.arc(lm.x, lm.y, radius, 0, 2 * Math.PI);
                ctx.fillStyle = isKeyJoint ? '#ffffff' : mainColor;
                ctx.fill();

                if (isKeyJoint) {
                  ctx.strokeStyle = mainColor;
                  ctx.lineWidth = 1.5;
                  ctx.stroke();
                }
              }
            });

            // Draw floating Knee/Hip Angle Badges near joints
            if (p.pose_angles) {
              const kneeAngle = p.pose_angles.left_knee ?? p.pose_angles.right_knee;
              if (kneeAngle !== undefined && kneeAngle !== null && lms[25]) {
                const kx = lms[25].x + 8;
                const ky = lms[25].y;
                ctx.font = 'bold 9px JetBrains Mono, monospace';
                const angleText = `∠ ${kneeAngle.toFixed(0)}°`;
                ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
                ctx.fillRect(kx, ky - 10, 42, 13);
                ctx.strokeStyle = '#10b981';
                ctx.strokeRect(kx, ky - 10, 42, 13);
                ctx.fillStyle = '#10b981';
                ctx.fillText(angleText, kx + 3, ky - 1);
              }
            }

            ctx.restore();
          }

          // B. Draw Interacting Objects & Connectors
          if (showObjects && p.interacting_objects && p.interacting_objects.length > 0) {
            const personCenterX = x + w / 2;
            const personCenterY = y + h / 2;

            p.interacting_objects.forEach((obj) => {
              const ox = obj.bbox.x * scaleX;
              const oy = obj.bbox.y * scaleY;
              const ow = obj.bbox.width * scaleX;
              const oh = obj.bbox.height * scaleY;
              const objCenterX = ox + ow / 2;
              const objCenterY = oy + oh / 2;

              const colorCfg = CATEGORY_COLORS[obj.category] || CATEGORY_COLORS.general_object;

              ctx.save();
              ctx.beginPath();
              ctx.moveTo(personCenterX, personCenterY);
              ctx.lineTo(objCenterX, objCenterY);
              ctx.strokeStyle = colorCfg.stroke;
              ctx.lineWidth = 1.5;
              ctx.setLineDash([4, 4]);
              ctx.stroke();

              ctx.setLineDash([]);
              ctx.fillStyle = colorCfg.fill;
              ctx.fillRect(ox, oy, ow, oh);
              ctx.strokeStyle = colorCfg.stroke;
              ctx.lineWidth = 1.5;
              ctx.strokeRect(ox, oy, ow, oh);

              ctx.font = 'bold 10px JetBrains Mono, monospace';
              const itemText = `◈ ${obj.label.toUpperCase()}`;
              const itemWidth = ctx.measureText(itemText).width + 8;
              ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
              ctx.fillRect(ox, Math.max(0, oy - 16), itemWidth, 16);
              ctx.strokeStyle = colorCfg.stroke;
              ctx.strokeRect(ox, Math.max(0, oy - 16), itemWidth, 16);
              ctx.fillStyle = colorCfg.stroke;
              ctx.fillText(itemText, ox + 4, Math.max(0, oy - 16) + 12);
              ctx.restore();
            });
          }

          // C. Draw Tactical HUD Bounding Box
          if (showBoundingBoxes) {
            ctx.save();
            ctx.lineWidth = isThreat ? 2.5 : 1.8;
            ctx.strokeStyle = mainColor;

            // Tactical Corner Brackets
            const cornerLen = Math.min(18, w * 0.25, h * 0.25);

            // Top-Left
            ctx.beginPath();
            ctx.moveTo(x, y + cornerLen);
            ctx.lineTo(x, y);
            ctx.lineTo(x + cornerLen, y);
            ctx.stroke();

            // Top-Right
            ctx.beginPath();
            ctx.moveTo(x + w - cornerLen, y);
            ctx.lineTo(x + w, y);
            ctx.lineTo(x + w, y + cornerLen);
            ctx.stroke();

            // Bottom-Left
            ctx.beginPath();
            ctx.moveTo(x, y + h - cornerLen);
            ctx.lineTo(x, y + h);
            ctx.lineTo(x + cornerLen, y + h);
            ctx.stroke();

            // Bottom-Right
            ctx.beginPath();
            ctx.moveTo(x + w - cornerLen, y + h);
            ctx.lineTo(x + w, y + h);
            ctx.lineTo(x + w, y + h - cornerLen);
            ctx.stroke();

            // Light inner tint
            ctx.fillStyle = isThreat ? 'rgba(239, 68, 68, 0.08)' : 'rgba(0, 240, 255, 0.04)';
            ctx.fillRect(x, y, w, h);

            // D. Header Info Tag Above Box
            if (showTrackingIds || showActivityLabels) {
              const trackText = showTrackingIds ? `#${p.track_id < 10 ? '0' + p.track_id : p.track_id}` : '';
              const actText = showActivityLabels ? `${p.activity.toUpperCase()} ${p.confidence ? p.confidence.toFixed(0) + '%' : ''}` : '';
              const fullLabel = [trackText, actText].filter(Boolean).join(' • ');

              ctx.font = 'bold 11px JetBrains Mono, monospace';
              const textWidth = ctx.measureText(fullLabel).width;
              const tagHeight = 20;

              const tagY = Math.max(0, y - tagHeight - 3);
              ctx.fillStyle = isThreat ? 'rgba(220, 38, 38, 0.95)' : 'rgba(15, 23, 42, 0.92)';
              ctx.fillRect(x, tagY, textWidth + 14, tagHeight);
              ctx.strokeStyle = mainColor;
              ctx.strokeRect(x, tagY, textWidth + 14, tagHeight);

              ctx.fillStyle = isThreat ? '#ffffff' : mainColor;
              ctx.fillText(fullLabel, x + 7, tagY + 14);
            }

            // E. ONLY for true dangerous threats, render warning banner
            if (isThreat && highlightThreats) {
              const warningText = isFighting ? '⚠ CRITICAL: VIOLENCE DETECTED' : '⚠ WEAPON DETECTED';
              ctx.font = 'bold 11px JetBrains Mono, monospace';
              const warnWidth = ctx.measureText(warningText).width;
              
              ctx.fillStyle = 'rgba(239, 68, 68, 0.95)';
              ctx.fillRect(x, y + h + 4, warnWidth + 12, 20);
              ctx.fillStyle = '#ffffff';
              ctx.fillText(warningText, x + 6, y + h + 18);
            }

            ctx.restore();
          }
        });
      }

      animFrameRef.current = requestAnimationFrame(render);
    };

    animFrameRef.current = requestAnimationFrame(render);

    return () => {
      active = false;
      if (animFrameRef.current !== null) {
        cancelAnimationFrame(animFrameRef.current);
      }
    };
  }, [people, objects, zones, showBoundingBoxes, showTrackingIds, showActivityLabels, showObjects, showZones, showSkeleton, videoWidth, videoHeight, highlightThreats]);

  return (
    <canvas
      ref={canvasRef}
      width={videoWidth}
      height={videoHeight}
      className="absolute inset-0 w-full h-full pointer-events-none z-20"
    />
  );
};
