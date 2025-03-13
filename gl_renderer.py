"""
Optimized renderer for the radio wave simulation using direct OpenGL.
"""
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math
import os
from PIL import Image
import glm

class ShaderProgram:
    """
    Utility class for creating and managing OpenGL shader programs.
    """
    def __init__(self):
        self.program_id = glCreateProgram()
        self.shaders = []
    
    def add_shader(self, shader_type, source):
        """Compile and attach a shader to the program."""
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source)
        glCompileShader(shader)
        
        # Check for compile errors
        if not glGetShaderiv(shader, GL_COMPILE_STATUS):
            error = glGetShaderInfoLog(shader).decode()
            raise RuntimeError(f"Shader compilation error: {error}")
        
        glAttachShader(self.program_id, shader)
        self.shaders.append(shader)
        
        return shader
    
    def link(self):
        """Link the shader program."""
        glLinkProgram(self.program_id)
        
        # Check for linking errors
        if not glGetProgramiv(self.program_id, GL_LINK_STATUS):
            error = glGetProgramInfoLog(self.program_id).decode()
            raise RuntimeError(f"Shader program linking error: {error}")
        
        # Once linked, we can detach and delete individual shaders
        for shader in self.shaders:
            glDetachShader(self.program_id, shader)
            glDeleteShader(shader)
        self.shaders = []
    
    def use(self):
        """Use this shader program."""
        glUseProgram(self.program_id)
    
    def set_uniform_1f(self, name, value):
        """Set a float uniform value."""
        location = glGetUniformLocation(self.program_id, name)
        glUniform1f(location, value)
    
    def set_uniform_2f(self, name, x, y):
        """Set a vec2 uniform value."""
        location = glGetUniformLocation(self.program_id, name)
        glUniform2f(location, x, y)
    
    def set_uniform_3f(self, name, x, y, z):
        """Set a vec3 uniform value."""
        location = glGetUniformLocation(self.program_id, name)
        glUniform3f(location, x, y, z)
    
    def set_uniform_4f(self, name, x, y, z, w):
        """Set a vec4 uniform value."""
        location = glGetUniformLocation(self.program_id, name)
        glUniform4f(location, x, y, z, w)
    
    def set_uniform_matrix4fv(self, name, matrix):
        """Set a mat4 uniform value."""
        location = glGetUniformLocation(self.program_id, name)
        glUniformMatrix4fv(location, 1, GL_FALSE, matrix)
    
    def set_uniform_1i(self, name, value):
        """Set an int uniform value."""
        location = glGetUniformLocation(self.program_id, name)
        glUniform1i(location, value)
    
    def cleanup(self):
        """Delete the shader program."""
        if self.program_id:
            glDeleteProgram(self.program_id)

class RadioWaveRenderer:
    """
    Pure OpenGL renderer for stars in the galaxy.
    """
    def __init__(self, width=800, height=600):
        """Initialize the renderer."""
        self.stars_data = []
        self.waves_data = []
        self.rotation_x = 350
        self.rotation_y = 340
        self.scale = 0.07
        self.translation_z = -1200
        
        # Window dimensions
        self.width = width
        self.height = height
        
        # Star sizes for better visibility
        self.normal_star_size = 24.0  # Base size for normal stars
        self.earth_star_size = 512.0   # Size for Earth
        self.active_earth_size = 1024.0 # Size for active Earth
        
        # Z-depth visualization parameters
        self.min_opacity = 0.1  # Reduced minimum opacity for better additive blending
        self.max_opacity = 0.7  # Reduced maximum opacity for better additive blending
        self.earth_opacity = 0.8
        self.size_min_scale = 1  # Size scale for distant stars
        self.size_max_scale = 15.0  # Size scale for close stars
        self.z_depth_factor = 2.0  # Power curve for depth effect
        self.z_depth_divisor = 8000  # Divisor for depth calculation
        
        # Initialize OpenGL state
        self.initialized = False
        self.star_texture_id = None
        
        # Debug info
        self.debug_wave_count = 0
        self.debug_burst_count = 0
        
        # Shader program for stars
        self.star_shader = None
        
        # Buffers for star rendering
        self.star_vao = None
        self.star_vbo = None
        self.star_instance_vbo = None
        self.star_instance_data = None
        
        # Star attributes (per instance)
        self.star_position_offset = 0
        self.star_color_offset = 12  # 3 floats (position) * 4 bytes
        self.star_size_offset = 28   # 3 floats (position) + 4 floats (color) * 4 bytes
        self.star_special_offset = 32  # 3 floats (position) + 4 floats (color) + 1 float (size) * 4 bytes
        self.star_stride = 36  # Total bytes per instance: 9 floats * 4 bytes
        
        # Add new shader and VAOs for wave rendering
        self.wave_shader = None
        self.wave_vao = None
        self.wave_vbo = None
        self.wave_circle_points = 48  # Number of points for circle approximation

        # Add new buffers for static and dynamic stars
        self.static_stars_vbo = None
        self.earth_star_vbo = None
        self.static_instance_count = 0
        self.earth_index = -1
        self.static_data_initialized = False

        # Add post-processing resources
        self.main_fbo = None
        self.bright_fbo = None
        self.blur_fbos = None
        self.screen_quad_vao = None
        self.post_shader = None
        self.blur_shader = None
        self.bright_shader = None
    
    def _create_star_shader(self):
        """Create shader program for rendering stars."""
        # Vertex shader with billboarding and view-space calculations
        vertex_shader = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec2 aTexCoord;
        layout (location = 2) in vec3 aInstancePos;
        layout (location = 3) in vec4 aInstanceColor;
        layout (location = 4) in float aInstanceSize;
        layout (location = 5) in float aInstanceSpecial;
        
        out vec2 TexCoord;
        out vec4 Color;
        out float IsEarthStar;
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        uniform float zDepthDivisor;
        uniform float zDepthFactor;
        uniform float minOpacity;
        uniform float maxOpacity;
        uniform float earthOpacity;
        uniform float sizeMinScale;
        uniform float sizeMaxScale;
        
        void main()
        {
            TexCoord = aTexCoord;
            IsEarthStar = aInstanceSpecial;
            
            // Transform instance position to view space for z-depth and billboarding
            vec4 viewPos = view * model * vec4(aInstancePos, 1.0);
            float viewZ = viewPos.z;
            
            // Calculate z-based scaling factor
            float zFactor = 1.0 - (viewZ / (zDepthDivisor + abs(viewPos.z)));
            zFactor = pow(max(0.0, min(1.0, zFactor)), zDepthFactor);
            
            // Calculate final size and opacity
            float size = aInstanceSize * max(sizeMinScale, min(sizeMaxScale, zFactor * 2.0))/4;
            float opacity = mix(minOpacity, maxOpacity, zFactor);
            
            // Earth gets special treatment - always high opacity
            if (aInstanceSpecial > 0.5) {
                opacity = earthOpacity;
            }
            
            // Apply glow boost for normal stars
            vec3 boostedColor = min(vec3(1.0), aInstanceColor.rgb * 1.2);
            Color = vec4(boostedColor, opacity);
            
            // Billboard calculation - cancel out rotation from view matrix
            // Extract the view matrix rotation without translation
            mat3 viewRotation = mat3(view);
            
            // Calculate billboard vertex offset in view space
            vec3 vertexOffset = vec3(aPos.x, aPos.y, 0.0) * size;
            
            // Apply billboard rotation and add to instance position
            vec3 worldPos = aInstancePos + (inverse(viewRotation) * vertexOffset);
            
            // Transform to clip space
            gl_Position = projection * view * model * vec4(worldPos, 1.0);
        }
        """
        
        # Fragment shader with proper alpha handling
        fragment_shader = """
        #version 330 core
        in vec2 TexCoord;
        in vec4 Color;
        in float IsEarthStar;
        
        out vec4 FragColor;
        
        uniform sampler2D starTexture;
        
        void main()
        {
            vec4 texColor = texture(starTexture, TexCoord);
            vec4 finalColor;
            
            if (IsEarthStar > 0.5) {
                // Special blending for Earth
                vec3 earthColor = Color.rgb * 2.0;  // Boost the base color
                
                // Add a bright blue glow
                float glowStrength = 1.0 - length(TexCoord - vec2(0.5));
                glowStrength = pow(max(0.0, glowStrength), 1.5);  // Adjust glow falloff
                vec3 glowColor = vec3(0.3, 0.5, 1.0);  // Bright blue glow
                
                finalColor = texColor * vec4(mix(earthColor, glowColor, glowStrength), Color.a);
                
                // Add extra brightness to the core
                float coreBrightness = 1.0 - length(TexCoord - vec2(0.5));
                coreBrightness = pow(max(0.0, coreBrightness), 2.0);
                finalColor.rgb += glowColor * coreBrightness;
            } else {
                // Normal star blending
                finalColor = texColor * Color;
            }
            
            // Discard nearly transparent pixels
            if (finalColor.a < 0.2)
                discard;
            
            // Simple energy-conserving bloom effect for normal stars
            if (IsEarthStar < 0.5) {
                float brightness = dot(finalColor.rgb, vec3(0.2126, 0.7152, 0.0722));
                if(brightness > 0.8)
                    finalColor.rgb *= 1.2;
            }
            
            FragColor = finalColor;
        }
        """
        
        # Create shader program
        shader_program = ShaderProgram()
        shader_program.add_shader(GL_VERTEX_SHADER, vertex_shader)
        shader_program.add_shader(GL_FRAGMENT_SHADER, fragment_shader)
        shader_program.link()
        
        return shader_program
    
    def _create_star_buffers(self):
        """Create buffers for star rendering with instancing."""
        # Create geometry buffers
        vertices = np.array([
            # positions        # texture coords
            -0.5, -0.5, 0.0,   0.0, 0.0,
             0.5, -0.5, 0.0,   1.0, 0.0,
             0.5,  0.5, 0.0,   1.0, 1.0,
            -0.5,  0.5, 0.0,   0.0, 1.0
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        
        # Create VBOs
        self.star_vbo = glGenBuffers(1)  # Geometry VBO
        self.static_stars_vbo = glGenBuffers(1)  # Static stars instance VBO
        self.earth_star_vbo = glGenBuffers(1)    # Earth instance VBO
        
        # Create EBO
        self.star_ebo = glGenBuffers(1)
        
        # Upload geometry data
        glBindBuffer(GL_ARRAY_BUFFER, self.star_vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.star_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        # Create VAO for static stars
        self.static_stars_vao = glGenVertexArrays(1)
        glBindVertexArray(self.static_stars_vao)
        
        # Bind geometry buffers and set attributes for static stars
        glBindBuffer(GL_ARRAY_BUFFER, self.star_vbo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.star_ebo)
        
        # Position attribute
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))
        
        # Texture coordinate attribute
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(3 * 4))
        
        # Bind instance VBO and set attributes for static stars
        glBindBuffer(GL_ARRAY_BUFFER, self.static_stars_vbo)
        
        # Instance attributes for static stars
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, self.star_stride, ctypes.c_void_p(self.star_position_offset))
        glVertexAttribDivisor(2, 1)
        
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, self.star_stride, ctypes.c_void_p(self.star_color_offset))
        glVertexAttribDivisor(3, 1)
        
        glEnableVertexAttribArray(4)
        glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, self.star_stride, ctypes.c_void_p(self.star_size_offset))
        glVertexAttribDivisor(4, 1)
        
        glEnableVertexAttribArray(5)
        glVertexAttribPointer(5, 1, GL_FLOAT, GL_FALSE, self.star_stride, ctypes.c_void_p(self.star_special_offset))
        glVertexAttribDivisor(5, 1)
        
        # Create separate VAO for Earth
        self.earth_star_vao = glGenVertexArrays(1)
        glBindVertexArray(self.earth_star_vao)
        
        # Bind geometry buffers and set attributes for Earth
        glBindBuffer(GL_ARRAY_BUFFER, self.star_vbo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.star_ebo)
        
        # Position attribute
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))
        
        # Texture coordinate attribute
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(3 * 4))
        
        # Bind instance VBO and set attributes for Earth
        glBindBuffer(GL_ARRAY_BUFFER, self.earth_star_vbo)
        
        # Instance attributes for Earth
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, self.star_stride, ctypes.c_void_p(self.star_position_offset))
        glVertexAttribDivisor(2, 1)
        
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, self.star_stride, ctypes.c_void_p(self.star_color_offset))
        glVertexAttribDivisor(3, 1)
        
        glEnableVertexAttribArray(4)
        glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, self.star_stride, ctypes.c_void_p(self.star_size_offset))
        glVertexAttribDivisor(4, 1)
        
        glEnableVertexAttribArray(5)
        glVertexAttribPointer(5, 1, GL_FLOAT, GL_FALSE, self.star_stride, ctypes.c_void_p(self.star_special_offset))
        glVertexAttribDivisor(5, 1)
        
        # Unbind VAO
        glBindVertexArray(0)
        
        print("Star buffers initialized")
    
    def initialize(self):
        """Initialize OpenGL state and load textures."""
        if self.initialized:
            return
            
        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Create star shader
        self.star_shader = self._create_star_shader()
        
        # Create star buffers
        self._create_star_buffers()
        
        # Create wave rendering resources
        self._create_wave_shader()
        self._create_wave_buffers()
        
        # Load the star texture
        self.star_texture_id = self._load_star_texture()
        
        # Initialize post-processing resources
        self._create_framebuffers()
        self._create_post_shaders()
        self._create_screen_quad()
        
        self.initialized = True

    def _create_framebuffers(self):
        """Create framebuffers for post-processing."""
        # Main scene FBO
        self.main_fbo = glGenFramebuffers(1)
        self.main_color_texture = glGenTextures(1)  # Changed name for clarity
        main_depth_buffer = glGenRenderbuffers(1)
        
        glBindFramebuffer(GL_FRAMEBUFFER, self.main_fbo)
        
        # Color attachment
        glBindTexture(GL_TEXTURE_2D, self.main_color_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, self.width, self.height, 0, GL_RGBA, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.main_color_texture, 0)
        
        # Depth attachment
        glBindRenderbuffer(GL_RENDERBUFFER, main_depth_buffer)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, self.width, self.height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, main_depth_buffer)
        
        # Verify main FBO is complete
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("Main framebuffer is not complete!")

        # Bright pass FBO
        self.bright_fbo = glGenFramebuffers(1)
        self.bright_color_texture = glGenTextures(1)  # Changed name for clarity
        
        glBindFramebuffer(GL_FRAMEBUFFER, self.bright_fbo)
        glBindTexture(GL_TEXTURE_2D, self.bright_color_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, self.width, self.height, 0, GL_RGBA, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.bright_color_texture, 0)
        
        # Verify bright FBO is complete
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("Bright framebuffer is not complete!")

        # Blur FBOs
        self.blur_fbos = []
        for i in range(2):
            blur_fbo = glGenFramebuffers(1)
            blur_texture = glGenTextures(1)
            
            glBindFramebuffer(GL_FRAMEBUFFER, blur_fbo)
            glBindTexture(GL_TEXTURE_2D, blur_texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, self.width, self.height, 0, GL_RGBA, GL_FLOAT, None)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, blur_texture, 0)
            
            # Verify blur FBO is complete
            if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
                raise RuntimeError(f"Blur framebuffer {i} is not complete!")
                
            self.blur_fbos.append((blur_fbo, blur_texture))
        
        # Reset to default framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def _load_star_texture(self):
        """Load the star texture from the PNG file.""" 
        # Get the path to the star texture
        current_dir = os.path.dirname(os.path.abspath(__file__))
        texture_path = os.path.join(current_dir, 'star-32p.png')
        
        # Load the texture using PIL
        try:
            img = Image.open(texture_path)
            # Convert to RGBA if not already
            img = img.convert("RGBA")
            img_data = np.array(img)
            
            # Verify the image has an alpha channel
            if img_data.shape[2] != 4:
                raise ValueError("Star texture must have an alpha channel")
                
            print(f"Loaded star texture: {img.size}, mode={img.mode}")
        except Exception as e:
            print(f"Error loading texture: {e}")
            # Create a simple fallback texture - a circle with soft edges
            size = 32
            img_data = np.zeros((size, size, 4), dtype=np.uint8)
            center = size // 2
            radius = size // 2 - 1
            
            for y in range(size):
                for x in range(size):
                    dx = x - center
                    dy = y - center
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist <= radius:
                        # Create a soft falloff
                        alpha = int(255 * (1.0 - min(1.0, dist / radius)))
                        img_data[y, x] = [255, 255, 255, alpha]
            print("Using fallback star texture")
        
        # Create OpenGL texture
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        
        # Set texture parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        # Upload texture data
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 
                     img_data.shape[1], img_data.shape[0], 
                     0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        
        print("Star texture initialized")
        return texture_id
    
    def _create_wave_shader(self):
        """Create shader for rendering wave spheres."""
        # Vertex shader with lighting
        vertex_shader = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 aNormal;
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        uniform vec3 center;
        uniform float radius;
        
        out vec3 Normal;
        out vec3 FragPos;
        
        void main()
        {
            // Create a sphere from the input position
            vec3 spherePos = normalize(aPos) * radius;
            vec3 worldPos = center + spherePos;
            
            // Pass fragment position and normal to fragment shader
            FragPos = worldPos;
            Normal = normalize(aPos);  // Since we're using a unit sphere, position equals normal
            
            // Transform to clip space
            gl_Position = projection * view * vec4(worldPos, 1.0);
        }
        """
        
        # Fragment shader with glowing translucent effect
        fragment_shader = """
        #version 330 core
        in vec3 Normal;
        in vec3 FragPos;
        
        out vec4 FragColor;
        
        uniform vec4 color;
        uniform vec3 center;
        
        void main()
        {
            // Calculate rim lighting effect
            vec3 viewPos = vec3(0.0, 0.0, 0.0);  // Camera position in view space
            vec3 viewDir = normalize(viewPos - FragPos);
            float rim = 1.0 - max(dot(viewDir, Normal), 0.0);
            rim = pow(rim, 2.0);  // Adjust power for sharper rim
            
            // Create glowing sphere effect
            vec4 glowColor = vec4(color.rgb * 1.2, color.a);
            
            // Mix rim lighting with base color
            vec4 finalColor = mix(glowColor, vec4(glowColor.rgb * 2.0, glowColor.a), rim);
            
            // Add fresnel-like effect for more depth
            float fresnel = pow(1.0 - abs(dot(viewDir, Normal)), 2.0);
            finalColor.rgb += color.rgb * fresnel * 0.5;
            
            FragColor = finalColor;
        }
        """
        
        # Create and compile shaders
        shader_program = ShaderProgram()
        shader_program.add_shader(GL_VERTEX_SHADER, vertex_shader)
        shader_program.add_shader(GL_FRAGMENT_SHADER, fragment_shader)
        shader_program.link()
        
        self.wave_shader = shader_program

    def _create_wave_buffers(self):
        """Create buffers for rendering wave spheres."""
        # Generate UV sphere vertices and indices for solid sphere
        vertices = []
        indices = []
        
        stacks = 16  # Vertical divisions
        slices = 32  # Horizontal divisions
        
        # Generate vertices
        for stack in range(stacks + 1):
            phi = math.pi * stack / stacks  # Vertical angle (0 to π)
            for slice in range(slices + 1):
                theta = 2.0 * math.pi * slice / slices  # Horizontal angle (0 to 2π)
                
                # Convert spherical to Cartesian coordinates (unit sphere)
                x = math.sin(phi) * math.cos(theta)
                y = math.sin(phi) * math.sin(theta)
                z = math.cos(phi)
                
                # Position (which is also the normal for a unit sphere)
                vertices.extend([x, y, z])
        
        # Generate indices for triangle strips
        for stack in range(stacks):
            for slice in range(slices):
                # Calculate vertex indices
                v1 = stack * (slices + 1) + slice
                v2 = v1 + slices + 1
                v3 = v1 + 1
                v4 = v2 + 1
                
                # Add two triangles for each quad
                indices.extend([v1, v2, v3])  # First triangle
                indices.extend([v3, v2, v4])  # Second triangle
        
        # Convert to numpy arrays
        vertices = np.array(vertices, dtype=np.float32)
        indices = np.array(indices, dtype=np.uint32)
        
        # Create and bind VAO
        self.wave_vao = glGenVertexArrays(1)
        glBindVertexArray(self.wave_vao)
        
        # Create and bind VBO
        self.wave_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.wave_vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        
        # Create and bind EBO
        self.wave_ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.wave_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        # Position attribute (which is also used as normal)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))
        
        # Normal attribute (same as position for unit sphere)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))
        
        # Store the number of indices for drawing
        self.wave_index_count = len(indices)
        
        # Unbind VAO
        glBindVertexArray(0)
    def update_view_params(self, rotation_x, rotation_y, scale, translation_z):
        """Update view parameters.""" 
        self.rotation_x = rotation_x % 360
        self.rotation_y = rotation_y % 360
        self.scale = max(0.001, min(100.0, scale))
        #self.scale = scale
        self.translation_z = translation_z
        
    def update_stars_data(self, stars_data):
        """Update star data for rendering.""" 
        self.stars_data = stars_data
        
    def update_waves_data(self, waves_data):
        """Update wave data for rendering.""" 
        self.waves_data = waves_data
        self.debug_wave_count = len(waves_data)
    
    def update_debug_info(self, wave_count, burst_count):
        """Update debug information.""" 
        self.debug_wave_count = wave_count
        self.debug_burst_count = burst_count
    
    def resize(self, width, height):
        """Handle window resize.""" 
        self.width = width
        self.height = height
        
        # Update OpenGL viewport
        glViewport(0, 0, width, height)
        
        # Recreate framebuffers with new size if initialized
        if self.initialized:
            if hasattr(self, 'main_fbo'):
                self._create_framebuffers()  # Recreate all framebuffers with new size
    
    def _transform_point(self, point):
        """Apply view transformation to a point.""" 
        x, y, z = point
        
        # Convert angles to radians
        rx = math.radians(self.rotation_x)
        ry = math.radians(self.rotation_y)
        
        # First rotate around X axis
        y_rot = y * math.cos(rx) - z * math.sin(rx)
        z_rot = y * math.sin(rx) + z * math.cos(rx)
        
        # Then rotate around Y axis
        x_rot = x * math.cos(ry) + z_rot * math.sin(ry)
        z_rot_final = -x * math.sin(ry) + z_rot * math.cos(ry)
        
        # Apply translation
        z_final = z_rot_final + self.translation_z
        
        return [x_rot, y_rot, z_final]
    
    
    def draw_wave_circles(self):
        """Draw animated wave spheres using modern OpenGL."""
        if not self.wave_shader:
            return
            
        # Use wave shader
        self.wave_shader.use()
        
        # Create view and projection matrices
        view_matrix = glm.mat4(1.0)  # Identity
        view_matrix = glm.translate(view_matrix, glm.vec3(0, 0, self.translation_z))
        view_matrix = glm.rotate(view_matrix, glm.radians(self.rotation_x), glm.vec3(1, 0, 0))
        view_matrix = glm.rotate(view_matrix, glm.radians(self.rotation_y), glm.vec3(0, 1, 0))
        view_matrix = glm.scale(view_matrix, glm.vec3(self.scale, self.scale, self.scale))
        
        projection_matrix = glm.perspective(glm.radians(45.0), self.width / self.height, 0.1, 5000.0)
        
        # Set matrices in shader
        self.wave_shader.set_uniform_matrix4fv("view", glm.value_ptr(view_matrix))
        self.wave_shader.set_uniform_matrix4fv("projection", glm.value_ptr(projection_matrix))
        self.wave_shader.set_uniform_matrix4fv("model", glm.value_ptr(glm.mat4(1.0)))
        
        # Enable blending and depth test
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)  # Additive blending for glow effect
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)  # Enable face culling for better transparency
        glCullFace(GL_BACK)     # Cull back faces
        glDepthMask(GL_FALSE)   # Don't write to depth buffer for transparent objects
        
        # Sort waves by distance from camera for correct transparency
        camera_pos = glm.vec3(0, 0, -self.translation_z)
        sorted_waves = sorted(self.waves_data, 
                            key=lambda w: -glm.length(camera_pos - glm.vec3(*w['position'])))
        
        # Bind wave VAO
        glBindVertexArray(self.wave_vao)
        
        # Draw each wave sphere from back to front
        for wave in sorted_waves:
            pos = wave['position']
            radius = wave['radius']
            color = wave['color']
            
            # Set uniform values for this wave
            self.wave_shader.set_uniform_3f("center", pos[0], pos[1], pos[2])
            self.wave_shader.set_uniform_1f("radius", radius)
            self.wave_shader.set_uniform_4f("color", color[0], color[1], color[2], color[3])
            
            # Draw the wave as triangles
            glDrawElements(GL_TRIANGLES, self.wave_index_count, GL_UNSIGNED_INT, None)
        
        # Restore OpenGL state
        glDisable(GL_CULL_FACE)
        glDepthMask(GL_TRUE)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBindVertexArray(0)
    
    def _prepare_star_instance_data(self, stars_data):
        """Prepare instance data for star rendering."""
        if not self.static_data_initialized:
            # First time initialization - separate static stars and Earth
            static_stars = []
            earth_data = None
            earth_index = -1

            for i, star in enumerate(stars_data):
                position = star['position']
                color = star['color']
                is_earth = star.get('is_earth', False)
                
                if (is_earth):
                    earth_index = i
                    earth_data = np.array([
                        *position,                    # position (3)
                        *color,                       # color (4)
                        self.earth_star_size,        # size (1)
                        1.0                          # special flag (1)
                    ], dtype=np.float32)
                else:
                    static_stars.extend([
                        *position,                    # position (3)
                        *color,                       # color (4)
                        self.normal_star_size,       # size (1)
                        0.0                          # special flag (1)
                    ])

            # Upload static stars data
            if static_stars:
                static_data = np.array(static_stars, dtype=np.float32)
                glBindBuffer(GL_ARRAY_BUFFER, self.static_stars_vbo)
                glBufferData(GL_ARRAY_BUFFER, static_data.nbytes, static_data, GL_STATIC_DRAW)
                self.static_instance_count = len(static_stars) // 9

            # Upload initial Earth data
            if earth_data is not None:
                glBindBuffer(GL_ARRAY_BUFFER, self.earth_star_vbo)
                glBufferData(GL_ARRAY_BUFFER, earth_data.nbytes, earth_data, GL_DYNAMIC_DRAW)
                self.earth_index = earth_index

            self.static_data_initialized = True
            return
        
        # For subsequent frames, only update Earth data if needed
        if self.earth_index >= 0:
            earth_star = stars_data[self.earth_index]
            is_active = earth_star.get('is_civilization_active', False)
            
            earth_data = np.array([
                *earth_star['position'],            # position (3)
                *earth_star['color'],               # color (4)
                self.active_earth_size if is_active else self.earth_star_size,  # size (1)
                1.0                                # special flag (1)
            ], dtype=np.float32)
            
            glBindBuffer(GL_ARRAY_BUFFER, self.earth_star_vbo)
            glBufferData(GL_ARRAY_BUFFER, earth_data.nbytes, earth_data, GL_DYNAMIC_DRAW)

    def draw(self):
        """Draw stars with post-processing effects."""
        if not self.initialized:
            self.initialize()
        
        # Clear with pure black background
        glClearColor(0.0, 0.0, 0.0, 1.0)
        
        # First pass: Render scene to main FBO
        glBindFramebuffer(GL_FRAMEBUFFER, self.main_fbo)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self._draw_scene()
        
        # Second pass: Extract bright areas with lower threshold for better bloom
        glBindFramebuffer(GL_FRAMEBUFFER, self.bright_fbo)
        glClear(GL_COLOR_BUFFER_BIT)
        self._apply_bright_pass()
        
        # Third pass: Apply gaussian blur with larger kernel
        self._apply_blur()
        
        # Final pass: Composite everything with enhanced effects
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glClear(GL_COLOR_BUFFER_BIT)
        self._apply_post_processing()

    def _draw_scene(self):
        """Draw the main scene (stars and waves)."""
        # Move existing draw logic here
        # Skip rendering if window is minimized or has invalid dimensions
        if self.width <= 0 or self.height <= 0:
            return
            
        # Clear the screen
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Prepare instance data (now only updates Earth when needed)
        self._prepare_star_instance_data(self.stars_data)
        
        # Enable blending with premultiplied alpha
        #glEnable(GL_BLEND)
        #glBlendEquation(GL_FUNC_ADD)
        #glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        
        # Set up matrices
        model_matrix = glm.mat4(1.0)  # Identity
        projection_matrix = glm.perspective(glm.radians(45.0), self.width / self.height, 0.1, 5000.0)
        
        # Create orbiting view matrix
        view_matrix = glm.mat4(1.0)
        view_matrix = glm.translate(view_matrix, glm.vec3(0, 0, self.translation_z))
        view_matrix = glm.rotate(view_matrix, glm.radians(self.rotation_x), glm.vec3(1, 0, 0))
        view_matrix = glm.rotate(view_matrix, glm.radians(self.rotation_y), glm.vec3(0, 1, 0))
        view_matrix = glm.scale(view_matrix, glm.vec3(self.scale, self.scale, self.scale))
        
        # Set up shader and uniforms
        self.star_shader.use()
        
        # Bind star texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.star_texture_id)
        self.star_shader.set_uniform_1i("starTexture", 0)
        
        # Set transform uniforms
        self.star_shader.set_uniform_matrix4fv("model", glm.value_ptr(model_matrix))
        self.star_shader.set_uniform_matrix4fv("view", glm.value_ptr(view_matrix))
        self.star_shader.set_uniform_matrix4fv("projection", glm.value_ptr(projection_matrix))
        
        # Set appearance uniforms
        self.star_shader.set_uniform_1f("zDepthDivisor", self.z_depth_divisor)
        self.star_shader.set_uniform_1f("zDepthFactor", self.z_depth_factor)
        self.star_shader.set_uniform_1f("minOpacity", self.min_opacity)
        self.star_shader.set_uniform_1f("maxOpacity", self.max_opacity)
        self.star_shader.set_uniform_1f("earthOpacity", self.earth_opacity)
        self.star_shader.set_uniform_1f("sizeMinScale", self.size_min_scale)
        self.star_shader.set_uniform_1f("sizeMaxScale", self.size_max_scale)
        
        # Draw static stars first
        if self.static_instance_count > 0:
            glBindVertexArray(self.static_stars_vao)
            glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None, self.static_instance_count)
        
        # Then draw Earth if it exists
        if self.earth_index >= 0:
            glBindVertexArray(self.earth_star_vao)
            glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None, 1)
        
        # Cleanup state
        glBindVertexArray(0)
        
        # Restore normal blending for UI and waves
        glBlendEquation(GL_FUNC_ADD)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Draw waves and debug info
        self.draw_wave_circles()

    def _apply_bright_pass(self):
        """Extract bright areas for bloom with improved parameters."""
        self.bright_shader.use()
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.main_color_texture)
        self.bright_shader.set_uniform_1i("scene", 0)
        self.bright_shader.set_uniform_1f("threshold", 0.7)  # Lower threshold for more bloom
        
        glDisable(GL_DEPTH_TEST)
        glBindVertexArray(self.screen_quad_vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)

    def _apply_blur(self):
        """Apply gaussian blur to the bright areas."""
        self.blur_shader.use()
        
        # Horizontal blur
        glBindFramebuffer(GL_FRAMEBUFFER, self.blur_fbos[0][0])
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.bright_color_texture)  # Use bright color texture
        self.blur_shader.set_uniform_1i("image", 0)
        self.blur_shader.set_uniform_2f("direction", 1.0, 0.0)
        glBindVertexArray(self.screen_quad_vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        
        # Vertical blur
        glBindFramebuffer(GL_FRAMEBUFFER, self.blur_fbos[1][0])
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.blur_fbos[0][1])
        self.blur_shader.set_uniform_2f("direction", 0.0, 1.0)
        glDrawArrays(GL_TRIANGLES, 0, 6)

    def _apply_post_processing(self):
        """Apply final post-processing with enhanced visual effects."""
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        
        self.post_shader.use()
        
        # Bind textures
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.main_color_texture)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.blur_fbos[1][1])
        
        # Enhanced post-processing parameters
        self.post_shader.set_uniform_1i("scene", 0)
        self.post_shader.set_uniform_1i("bloom", 1)
        self.post_shader.set_uniform_1f("exposure", 1.0)  # Increased exposure
        self.post_shader.set_uniform_1f("gamma", 1.6)  # Proper gamma correction
        
        # Update post shader with enhanced color grading
        self.post_shader.set_uniform_3f("tint", 1.1, 1.05, 1.0)  # Slight yellow tint
        self.post_shader.set_uniform_1f("saturation", 1.2)  # Enhanced colors
        self.post_shader.set_uniform_1f("contrast", 1.1)  # Increased contrast
        
        glDisable(GL_DEPTH_TEST)
        glBindVertexArray(self.screen_quad_vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glEnable(GL_DEPTH_TEST)

    def _create_post_shaders(self):
        """Create shaders for post-processing effects."""
        # Bright pass shader for bloom
        bright_vs = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec2 aTexCoords;
        
        out vec2 TexCoords;
        
        void main() {
            gl_Position = vec4(aPos, 1.0);
            TexCoords = aTexCoords;
        }
        """
        
        bright_fs = """
        #version 330 core
        in vec2 TexCoords;
        out vec4 FragColor;
        
        uniform sampler2D scene;
        uniform float threshold;
        
        void main() {
            vec3 color = texture(scene, TexCoords).rgb;
            float brightness = dot(color, vec3(0.2126, 0.7152, 0.0722));
            if(brightness > threshold)
                FragColor = vec4(color, 1.0);
            else
                FragColor = vec4(0.0, 0.0, 0.0, 1.0);
        }
        """
        
        # Gaussian blur shader
        blur_vs = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec2 aTexCoords;
        
        out vec2 TexCoords;
        
        void main() {
            gl_Position = vec4(aPos, 1.0);
            TexCoords = aTexCoords;
        }
        """
        
        blur_fs = """
        #version 330 core
        in vec2 TexCoords;
        out vec4 FragColor;
        
        uniform sampler2D image;
        uniform vec2 direction;
        
        void main() {
            vec2 tex_offset = 1.0 / textureSize(image, 0);
            float weights[5] = float[] (0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
            
            vec3 result = texture(image, TexCoords).rgb * weights[0];
            for(int i = 1; i < 5; ++i) {
                result += texture(image, TexCoords + direction * tex_offset * i).rgb * weights[i];
                result += texture(image, TexCoords - direction * tex_offset * i).rgb * weights[i];
            }
            FragColor = vec4(result, 1.0);
        }
        """
        
        # Final post-processing shader
        post_vs = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec2 aTexCoords;
        
        out vec2 TexCoords;
        
        void main() {
            gl_Position = vec4(aPos, 1.0);
            TexCoords = aTexCoords;
        }
        """
        
        post_fs = """
    #version 330 core
    in vec2 TexCoords;
    out vec4 FragColor;
    
    uniform sampler2D scene;
    uniform sampler2D bloom;
    uniform float exposure;
    uniform float gamma;
    uniform vec3 tint;
    uniform float saturation;
    uniform float contrast;
    
    vec3 adjustSaturation(vec3 color, float adjustment)
    {
        const vec3 luminance = vec3(0.2126, 0.7152, 0.0722);
        float luminanceValue = dot(color, luminance);
        return mix(vec3(luminanceValue), color, adjustment);
    }
    
    void main()
    {
        // Sample textures
        vec3 hdrColor = texture(scene, TexCoords).rgb;
        vec3 bloomColor = texture(bloom, TexCoords).rgb;
        
        // Add bloom with higher intensity
        vec3 result = hdrColor + bloomColor * 1.5;       
        
        // Tone mapping with enhanced contrast
        result = vec3(1.0) - exp(-result * exposure);
        
        // Color grading
        result = adjustSaturation(result, saturation);
        result = pow(result * tint, vec3(1.0 / gamma));
        
        // Contrast adjustment
        result = mix(vec3(0.5), result, contrast);
        
        FragColor = vec4(result, 1.0);
    }
"""
        
        # Create shader programs
        self.bright_shader = ShaderProgram()
        self.bright_shader.add_shader(GL_VERTEX_SHADER, bright_vs)
        self.bright_shader.add_shader(GL_FRAGMENT_SHADER, bright_fs)
        self.bright_shader.link()
        
        self.blur_shader = ShaderProgram()
        self.blur_shader.add_shader(GL_VERTEX_SHADER, blur_vs)
        self.blur_shader.add_shader(GL_FRAGMENT_SHADER, blur_fs)
        self.blur_shader.link()
        
        self.post_shader = ShaderProgram()
        self.post_shader.add_shader(GL_VERTEX_SHADER, post_vs)
        self.post_shader.add_shader(GL_FRAGMENT_SHADER, post_fs)
        self.post_shader.link()

    def _create_screen_quad(self):
        """Create a screen-space quad for post-processing."""
        vertices = np.array([
            # positions        # texture coords
            -1.0,  1.0, 0.0,  0.0, 1.0,
            -1.0, -1.0, 0.0,  0.0, 0.0,
             1.0, -1.0, 0.0,  1.0, 0.0,
             
            -1.0,  1.0, 0.0,  0.0, 1.0,
             1.0, -1.0, 0.0,  1.0, 0.0,
             1.0,  1.0, 0.0,  1.0, 1.0
        ], dtype=np.float32)
        
        self.screen_quad_vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        
        glBindVertexArray(self.screen_quad_vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(3 * 4))
        
        glBindVertexArray(0)
        glDeleteBuffers(1, [vbo])

    def cleanup(self):
        """Clean up OpenGL resources.""" 
        if self.star_texture_id is not None:
            glDeleteTextures(1, [self.star_texture_id])
            
        if self.star_shader is not None:
            self.star_shader.cleanup()
            
        if self.wave_shader is not None:
            self.wave_shader.cleanup()
            
        if self.star_vao is not None:
            glDeleteVertexArrays(1, [self.star_vao])
            
        if self.star_vbo is not None:
            glDeleteBuffers(1, [self.star_vbo])
            
        if self.star_instance_vbo is not None:
            glDeleteBuffers(1, [self.star_instance_vbo])
            
        if self.star_ebo is not None:
            glDeleteBuffers(1, [self.star_ebo])
            
        if self.wave_vao is not None:
            glDeleteVertexArrays(1, [self.wave_vao])
            
        if self.wave_vbo is not None:
            glDeleteBuffers(1, [self.wave_vbo])
        
        if self.static_stars_vbo is not None:
            glDeleteBuffers(1, [self.static_stars_vbo])
        
        if self.earth_star_vbo is not None:
            glDeleteBuffers(1, [self.earth_star_vbo])

        # Clean up post-processing resources
        if hasattr(self, 'main_fbo_texture'):
            glDeleteTextures(1, [self.main_fbo_texture])
        if hasattr(self, 'bright_fbo_texture'):
            glDeleteTextures(1, [self.bright_fbo_texture])
        if hasattr(self, 'main_fbo'):
            glDeleteFramebuffers(1, [self.main_fbo])
        if hasattr(self, 'bright_fbo'):
            glDeleteFramebuffers(1, [self.bright_fbo])
        if hasattr(self, 'blur_fbos'):
            for fbo, tex in self.blur_fbos:
                glDeleteFramebuffers(1, [fbo])
                glDeleteTextures(1, [tex])

