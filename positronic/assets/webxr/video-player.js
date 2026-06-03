/**
 * WebXR Video Player class that displays a video stream on a plane in AR
 */
export class WebXRVideoPlayer {
    constructor({ mode = "world", followCamera = true } = {}) {
        this.videoTexture = null;
        this.videoMaterial = null;
        this.videoPlane = null;
        this.videoGroup = null;
        this.screenContainer = null;
        this.screenImage = null;
        this.ws = null;
        this.scene = null;
        this.camera = null;
        this.mode = mode;
        this.followCamera = followCamera;
    }

    async init(scene, camera) {
        console.log("Initializing WebXRVideoPlayer");
        this.scene = scene;
        this.camera = camera;

        if (this.mode === "screen") {
            this.initScreenOverlay();
            await this.connectWebSocket();
            return;
        }

        // Create video texture and material
        this.videoTexture = new THREE.Texture();
        this.videoMaterial = new THREE.ShaderMaterial({
            uniforms: {
                map: { value: this.videoTexture },
                blueBoost: { value: 1.5 },
                contrast: { value: 2.0 }  // 1.0 is normal, >1.0 increases contrast
            },
            vertexShader: `
                varying vec2 vUv;
                void main() {
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform sampler2D map;
                uniform float blueBoost;
                uniform float contrast;
                varying vec2 vUv;
                void main() {
                    vec4 texColor = texture2D(map, vUv);
                    // Apply contrast adjustment
                    texColor.rgb = (texColor.rgb - 0.5) * contrast + 0.5;
                    // Boost blue channel
                    texColor.b = min(1.0, texColor.b * blueBoost);
                    // Clamp final colors to valid range
                    texColor.rgb = clamp(texColor.rgb, 0.0, 1.0);
                    gl_FragColor = texColor;
                }
            `,
            side: THREE.DoubleSide,
            transparent: true,
        });

        // Create plane for video (1 meter wide, 16:9 aspect ratio)
        const width = 1.0;
        const height = width * (9/16);
        this.videoPlane = new THREE.Mesh(
            new THREE.PlaneGeometry(width, height),
            this.videoMaterial
        );

        // this.videoPlane.rotation.z = -Math.PI / 2;

        // Create a group to hold the video plane
        this.videoGroup = new THREE.Group();
        this.videoGroup.add(this.videoPlane);
        this.scene.add(this.videoGroup);
        this.videoGroup.visible = true;


        // Connect to video websocket
        console.log("Connecting to video websocket...");
        await this.connectWebSocket();
    }

    initScreenOverlay() {
        this.screenContainer = document.createElement("div");
        this.screenContainer.style.position = "fixed";
        this.screenContainer.style.left = "50%";
        this.screenContainer.style.top = "42%";
        this.screenContainer.style.transform = "translate(-50%, -50%)";
        this.screenContainer.style.width = "min(78vw, calc(78vh * 16 / 9))";
        this.screenContainer.style.maxWidth = "720px";
        this.screenContainer.style.aspectRatio = "16 / 9";
        this.screenContainer.style.zIndex = "20";
        this.screenContainer.style.pointerEvents = "none";
        this.screenContainer.style.border = "2px solid rgba(255, 255, 255, 0.85)";
        this.screenContainer.style.borderRadius = "10px";
        this.screenContainer.style.overflow = "hidden";
        this.screenContainer.style.background = "rgba(0, 0, 0, 0.55)";
        this.screenContainer.style.boxShadow = "0 8px 24px rgba(0, 0, 0, 0.45)";

        this.screenImage = document.createElement("img");
        this.screenImage.style.display = "block";
        this.screenImage.style.width = "100%";
        this.screenImage.style.height = "100%";
        this.screenImage.style.objectFit = "cover";
        this.screenContainer.appendChild(this.screenImage);
        document.body.appendChild(this.screenContainer);
    }

    async connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/video`;

        this.ws = new WebSocket(wsUrl);

        this.ws.onmessage = async (event) => {
            try {
                const src = 'data:image/jpeg;base64,' + event.data;
                if (this.mode === "screen") {
                    this.screenImage.src = src;
                    return;
                }

                // Create an image from the base64 data
                const img = new Image();
                img.onload = () => {
                    this.videoTexture.image = img;
                    this.videoTexture.needsUpdate = true;
                };
                img.src = src;
            } catch (error) {
                console.error("Error processing video frame:", error);
            }
        };

        this.ws.onopen = () => {
            console.log("Video WebSocket connected");
        };

        this.ws.onerror = (error) => {
            console.error("Video WebSocket error:", error);
        };

        this.ws.onclose = () => {
            console.log("Video WebSocket closed, attempting to reconnect...");
            setTimeout(() => this.connectWebSocket(), 1000);
        };
    }

    update() {
        if (this.mode === "screen") return;
        if (!this.videoGroup || !this.camera || !this.videoGroup.visible) return;
        if (!this.followCamera) return;

        // Get camera position and orientation
        const cameraPosition = new THREE.Vector3();
        const cameraQuaternion = new THREE.Quaternion();
        this.camera.getWorldPosition(cameraPosition);
        this.camera.getWorldQuaternion(cameraQuaternion);

        // Position the video plane 2 meters in front of the camera
        const offset = new THREE.Vector3(0, 0, -2);
        offset.applyQuaternion(cameraQuaternion);

        // Update video plane position and orientation
        this.videoGroup.position.copy(cameraPosition).add(offset);
        this.videoGroup.quaternion.copy(cameraQuaternion);
    }

    show() {
        if (this.screenContainer) {
            this.screenContainer.style.display = "block";
            return;
        }
        if (this.videoGroup) {
            console.log("Showing video plane");
            this.videoGroup.visible = true;
        }
    }

    hide() {
        if (this.screenContainer) {
            this.screenContainer.style.display = "none";
            return;
        }
        if (this.videoGroup) {
            console.log("Hiding video plane");
            this.videoGroup.visible = false;
        }
    }

    dispose() {
        if (this.ws) {
            this.ws.close();
        }
        if (this.videoTexture) {
            this.videoTexture.dispose();
        }
        if (this.videoMaterial) {
            this.videoMaterial.dispose();
        }
        if (this.videoPlane) {
            this.videoPlane.geometry.dispose();
        }
        if (this.videoGroup && this.scene) {
            this.scene.remove(this.videoGroup);
        }
        if (this.screenContainer) {
            this.screenContainer.remove();
        }
    }
}
