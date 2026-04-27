"use client";

import { useEffect, useRef } from "react";

/**
 * Three.js dithered-surface visualisation. Decorative — port of the
 * shader from the v2 design reference. Lazy-imported so it doesn't
 * block first paint, and gracefully no-ops if WebGL isn't available
 * (the rest of the Home panel still renders as text).
 *
 * The shader is a low-frequency sin-field plane plus an ordered 4×4
 * Bayer dither in the fragment shader. We keep the geometry and shader
 * inline (no .glsl files) so SSR + hot-reload behave.
 */
export function DitheredSurface() {
  const mountRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    let cancelled = false;
    let cleanup: (() => void) | null = null;

    (async () => {
      let THREE: typeof import("three");
      try {
        THREE = await import("three");
      } catch {
        return;
      }
      if (cancelled || !mount) return;

      const width = mount.offsetWidth;
      const height = mount.offsetHeight;
      if (width === 0 || height === 0) return;

      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
      camera.position.set(1.5, 1.5, 1.5);
      camera.lookAt(0, 0, 0);

      let renderer: import("three").WebGLRenderer;
      try {
        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      } catch {
        return; // No WebGL.
      }
      // Read the live --surface CSS var so the WebGL clear color
      // tracks the theme (white in light, near-black in dark).
      // Falls back to white if the var isn't readable.
      const css = getComputedStyle(document.documentElement);
      const themedSurface = css.getPropertyValue("--surface").trim() || "#ffffff";
      renderer.setClearColor(new THREE.Color(themedSurface), 1);
      renderer.setSize(width, height);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      mount.appendChild(renderer.domElement);

      const VS = `
        varying vec2 vUv;
        varying float vElevation;
        uniform float uTime;
        void main() {
          vUv = uv;
          vec4 mp = modelMatrix * vec4(position, 1.0);
          float r = length(mp.xz);
          float e =
            sin(r * 4.0 - uTime * 1.2) * 0.18 +
            sin(mp.x * 3.0 + uTime) * sin(mp.z * 2.0 + uTime * 0.5) * 0.25 -
            exp(-r * r * 4.0) * 0.35;
          mp.y += e;
          vElevation = e;
          gl_Position = projectionMatrix * viewMatrix * mp;
        }
      `;

      // Ordered 4×4 Bayer dither. Flattens to two grayscale tones, so
      // the surface reads as black-and-white halftone — matches the
      // v2 design reference.
      const FS = `
        varying vec2 vUv;
        varying float vElevation;
        uniform vec3 uBgColor;
        uniform vec3 uFgColor;

        float dither(vec2 p, float b) {
          int x = int(mod(p.x, 4.0));
          int y = int(mod(p.y, 4.0));
          int idx = x + y * 4;
          float lim = 0.0;
          if (idx == 0)  lim = 0.0625;
          if (idx == 8)  lim = 0.5625;
          if (idx == 2)  lim = 0.1875;
          if (idx == 10) lim = 0.6875;
          if (idx == 12) lim = 0.8125;
          if (idx == 4)  lim = 0.3125;
          if (idx == 14) lim = 0.9375;
          if (idx == 6)  lim = 0.4375;
          if (idx == 3)  lim = 0.25;
          if (idx == 11) lim = 0.75;
          if (idx == 1)  lim = 0.125;
          if (idx == 9)  lim = 0.625;
          if (idx == 15) lim = 1.0;
          if (idx == 7)  lim = 0.5;
          if (idx == 13) lim = 0.875;
          if (idx == 5)  lim = 0.375;
          return b < lim ? 0.0 : 1.0;
        }

        void main() {
          float light = vElevation * 2.2 + 0.55;
          float scan = sin(vUv.y * 600.0) * 0.06;
          light += scan;
          vec2 sp = gl_FragCoord.xy / 2.0;
          float c = dither(sp, light);
          vec3 col = mix(uBgColor, uFgColor, c);
          gl_FragColor = vec4(col, 1.0);
        }
      `;

      const geo = new THREE.PlaneGeometry(3, 3, 128, 128);
      geo.rotateX(-Math.PI * 0.5);
      // Drive bg/fg colors from CSS so the dither tracks the theme.
      // Light: bg=0.97 (white-ish), fg=0.05 (near-black).
      // Dark:  bg=0.05 (near-black), fg=0.97 (white-ish).
      const isDark = themedSurface.startsWith("#0") ||
        themedSurface.startsWith("#1") ||
        themedSurface.toLowerCase().includes("0a0a") ||
        themedSurface.toLowerCase().includes("0b0b");
      const bg3 = isDark ? new THREE.Vector3(0.04, 0.04, 0.05) : new THREE.Vector3(0.97, 0.97, 0.97);
      const fg3 = isDark ? new THREE.Vector3(0.78, 0.78, 0.80) : new THREE.Vector3(0.05, 0.05, 0.05);
      const mat = new THREE.ShaderMaterial({
        vertexShader: VS,
        fragmentShader: FS,
        uniforms: {
          uTime: { value: 0 },
          uBgColor: { value: bg3 },
          uFgColor: { value: fg3 },
        },
        transparent: true,
      });
      const mesh = new THREE.Mesh(geo, mat);
      scene.add(mesh);

      const clock = new THREE.Clock();
      let mx = 0;
      let my = 0;
      let raf = 0;

      const animate = () => {
        const t = clock.getElapsedTime();
        mat.uniforms.uTime.value = t * 0.5;
        mesh.rotation.y = Math.sin(t * 0.2) * 0.1;
        camera.position.x += (mx * 0.5 + 1.5 - camera.position.x) * 0.04;
        camera.position.y += (my * 0.5 + 1.5 - camera.position.y) * 0.04;
        camera.lookAt(0, 0, 0);
        renderer.render(scene, camera);
        raf = requestAnimationFrame(animate);
      };
      raf = requestAnimationFrame(animate);

      const onResize = () => {
        if (!mount.offsetWidth) return;
        camera.aspect = mount.offsetWidth / mount.offsetHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(mount.offsetWidth, mount.offsetHeight);
      };
      window.addEventListener("resize", onResize);

      const onMouse = (e: MouseEvent) => {
        mx = e.clientX / window.innerWidth - 0.5;
        my = e.clientY / window.innerHeight - 0.5;
      };
      window.addEventListener("mousemove", onMouse);

      // Live theme tracking: re-read --surface and flip the dither
      // bg/fg + clear color whenever the data-theme attribute changes.
      // Without this, switching to dark mode leaves the surface
      // rendered in light-mode colors until a page reload.
      const themeObserver = new MutationObserver(() => {
        const cssNow = getComputedStyle(document.documentElement);
        const surfNow = cssNow.getPropertyValue("--surface").trim() || "#ffffff";
        const isDarkNow =
          surfNow.startsWith("#0") ||
          surfNow.startsWith("#1") ||
          surfNow.toLowerCase().includes("0a0a") ||
          surfNow.toLowerCase().includes("0b0b");
        renderer.setClearColor(new THREE.Color(surfNow), 1);
        const newBg = isDarkNow
          ? [0.04, 0.04, 0.05]
          : [0.97, 0.97, 0.97];
        const newFg = isDarkNow
          ? [0.78, 0.78, 0.80]
          : [0.05, 0.05, 0.05];
        const u = mat.uniforms.uBgColor.value;
        const v = mat.uniforms.uFgColor.value;
        u.set(newBg[0], newBg[1], newBg[2]);
        v.set(newFg[0], newFg[1], newFg[2]);
      });
      themeObserver.observe(document.documentElement, {
        attributes: true,
        attributeFilter: ["data-theme"],
      });

      cleanup = () => {
        cancelAnimationFrame(raf);
        window.removeEventListener("resize", onResize);
        window.removeEventListener("mousemove", onMouse);
        themeObserver.disconnect();
        renderer.dispose();
        geo.dispose();
        mat.dispose();
        if (renderer.domElement.parentNode) {
          renderer.domElement.parentNode.removeChild(renderer.domElement);
        }
      };
    })();

    return () => {
      cancelled = true;
      if (cleanup) cleanup();
    };
  }, []);

  return (
    <div
      ref={mountRef}
      className="absolute inset-0"
      aria-hidden="true"
      // The shader is decorative — text fallback already exists in the
      // parent. Hide from assistive tech.
    />
  );
}
