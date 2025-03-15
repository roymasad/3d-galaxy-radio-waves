"""
3D Galaxy Radio Wave Simulation using ImGui and OpenGL
"""
import os
import math
import time
import numpy as np
import glfw
import imgui
from imgui.integrations.glfw import GlfwRenderer
from OpenGL.GL import *
from OpenGL.GLU import *
import glm
import platform

from simulation import Simulation
from gl_renderer import RadioWaveRenderer

class RadioWaveSimApp:
    """
    Main application class for the Radio Wave Propagation Simulator.
    """
    def __init__(self, width=1280, height=800):
        """Initialize the application."""
        self.width = width
        self.height = height
        self.window = None
        self.imgui_renderer = None
        
        # Simulation parameters
        self.galaxy_diameter = 10000
        self.num_stars = 10000
        self.num_radio_emitting = 50
        self.simulation_speed = 1500
        self.simulation_duration = 100000
        self.radio_signal_duration = 10000
        self.radio_civilization_lifetime = 10000
        self.radio_blast_duration = 500
        self.sky_coverage_percent = 30
        
        # Create simulation
        self.simulation = Simulation(
            galaxy_diameter=self.galaxy_diameter,
            num_stars=self.num_stars,
            num_radio_emitting=self.num_radio_emitting,
            simulation_speed=self.simulation_speed,
            simulation_duration=self.simulation_duration,
            radio_signal_duration=self.radio_signal_duration,
            radio_civilization_lifetime=self.radio_civilization_lifetime,
            radio_blast_duration=self.radio_blast_duration,
            sky_coverage_percent=self.sky_coverage_percent
        )
        
        # Create renderer
        self.renderer = None
        
        # View parameters
        self.rotation_x = 350
        self.rotation_y = 340
        self.rotation_z = 0  # Add Z-axis rotation parameter
        self.scale = 0.07
        self.translation_z = -1200
        
        # Auto-rotation parameters
        self.auto_rotation_enabled = True
        self.auto_rotation_speed = 360.0 / 180.0  # 1 rotation each 3 minutes
        self.last_auto_rotation_time = time.time()
        
        # Timing
        self.last_update_time = time.time()
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
        
        # Mouse handling
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.mouse_pressed = False
        self.mouse_dragging = False
        
        # UI interaction state
        self.is_hovering_ui = False
        self.ui_rect = (10, 10, 410, 580)  # x, y, width, height of UI panel
        
        # Simulation state
        self.is_running = False
        
        # Text input buffers for ImGui
        self.text_buffers = {
            "galaxy_diameter": "10000",
            "num_stars": "10000",
            "num_radio_emitting": "50",
            "simulation_speed": "1500",
            "simulation_duration": "100000",
            "radio_signal_duration": "10000",
            "radio_civilization_lifetime": "10000",
            "radio_blast_duration": "500",
            "sky_coverage_percent": "30"
        }
        
            
    def get_system_mono_font_path(self):
        if platform.system() == "Windows":
            return "C:\\Windows\\Fonts\\consola.ttf"
        elif platform.system() == "Darwin":
            return "/System/Library/Fonts/Menlo.ttc"
        return None

    def init_glfw(self):
        """Initialize GLFW window and ImGui."""
        # Initialize GLFW
        if not glfw.init():
            print("Failed to initialize GLFW")
            return False
        
        # Configure GLFW
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)  # Required for Mac
        glfw.window_hint(glfw.SAMPLES, 4)  # Anti-aliasing
        glfw.window_hint(glfw.MAXIMIZED, GL_TRUE)  # Start maximized
        
        
        # Create window
        self.window = glfw.create_window(
            self.width, self.height, "Galactic Civilizations Radio Wave Detection Simulator 0.2 2025", None, None
        )
        
        if not self.window:
            print("Failed to create GLFW window")
            glfw.terminate()
            return False
        
        # Set window as current context
        glfw.make_context_current(self.window)
        
        # VSync
        glfw.swap_interval(1)
        
        # Initialize ImGui
        imgui.create_context()
        self.imgui_io = imgui.get_io()
        
        # Increase font size
        #font_size = 22  # Original size * 1.3
        self.imgui_io.font_global_scale = 1.5
        
        # Set custom scroll callback before creating ImGui renderer
        # This ensures our callback gets called first
        glfw.set_scroll_callback(self.window, self.scroll_callback)
        
        # Now create ImGui renderer - it will add its own callbacks
        self.imgui_renderer = GlfwRenderer(self.window)
        
        # Re-register our callbacks after ImGui initialization
        # to override ImGui's callbacks
        glfw.set_framebuffer_size_callback(self.window, self.framebuffer_size_callback)
        glfw.set_key_callback(self.window, self.key_callback)
        glfw.set_cursor_pos_callback(self.window, self.cursor_pos_callback)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        # Re-register scroll callback to ensure it's not overridden
        glfw.set_scroll_callback(self.window, self.scroll_callback)
        
        print("ImGui renderer created with custom callbacks")
        
        # Configure ImGui style
        style = imgui.get_style()
        style.window_padding = (10, 10)
        style.frame_padding = (5, 5)
        style.item_spacing = (8, 4)
        style.item_inner_spacing = (5, 5)
        style.alpha = 1.0
        style.window_rounding = 10.0
        style.frame_rounding = 5.0
        style.colors[imgui.COLOR_TEXT] = (1.0, 1.0, 1.0, 1.0)
        style.colors[imgui.COLOR_WINDOW_BACKGROUND] = (0.1, 0.1, 0.1, 0.9)
        style.colors[imgui.COLOR_FRAME_BACKGROUND] = (0.2, 0.2, 0.2, 1.0)
        style.colors[imgui.COLOR_FRAME_BACKGROUND_HOVERED] = (0.3, 0.3, 0.3, 1.0)
        style.colors[imgui.COLOR_FRAME_BACKGROUND_ACTIVE] = (0.4, 0.4, 0.4, 1.0)
        style.colors[imgui.COLOR_BUTTON] = (0.25, 0.25, 0.25, 1.0)
        style.colors[imgui.COLOR_BUTTON_HOVERED] = (0.35, 0.35, 0.35, 1.0)
        style.colors[imgui.COLOR_BUTTON_ACTIVE] = (0.45, 0.45, 0.45, 1.0)
        style.colors[imgui.COLOR_HEADER] = (0.3, 0.3, 0.3, 1.0)
        style.colors[imgui.COLOR_HEADER_HOVERED] = (0.4, 0.4, 0.4, 1.0)
        style.colors[imgui.COLOR_HEADER_ACTIVE] = (0.5, 0.5, 0.5, 1.0)
        
        # Initialize OpenGL
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Create renderer
        self.renderer = RadioWaveRenderer(width=self.width, height=self.height)
        
        # Set initial viewport
        width, height = glfw.get_framebuffer_size(self.window)
        self.renderer.resize(width, height)
        
        font_path = self.get_system_mono_font_path()
        self.label_font = ""
        
        if font_path:
            self.label_font = self.imgui_io.fonts.add_font_from_file_ttf(font_path, 28)
        self.imgui_renderer.refresh_font_texture()
                
        print("GLFW and ImGui initialized")
        return True
    
    def is_point_in_ui(self, x, y):
        """Check if a point is inside the UI panel area."""
        ui_x, ui_y, ui_width, ui_height = self.ui_rect
        return (ui_x <= x <= ui_x + ui_width) and (ui_y <= y <= ui_y + ui_height)
    
    def framebuffer_size_callback(self, window, width, height):
        """Handle window resize."""
        self.width = width
        self.height = height
        # Only update renderer if window has valid dimensions
        if width > 0 and height > 0:
            self.renderer.resize(width, height)
    
    def key_callback(self, window, key, scancode, action, mods):
        """Handle keyboard input."""
        # Let ImGui process key events first
        # If ImGui wants keyboard input (i.e., when entering text in a field), don't handle the event further
        if self.imgui_io.want_capture_keyboard:
            # Forward the event to imgui renderer
            self.imgui_renderer.keyboard_callback(window, key, scancode, action, mods)
            return
            
        # Handle our own events only if ImGui doesn't need the keyboard
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)
    
    def cursor_pos_callback(self, window, x, y):
        """Handle mouse movement."""
        # Check if the mouse is over the UI panel
        is_over_ui = self.is_point_in_ui(x, y)
        self.is_hovering_ui = is_over_ui
        
        # Manual handling of rotation
        if self.mouse_pressed and not (self.imgui_io.want_capture_mouse or is_over_ui):
            dx = x - self.last_mouse_x
            dy = y - self.last_mouse_y
            
            if abs(dx) > 0 or abs(dy) > 0:  # Only update if there's actual movement
                # Adjust rotation with sensitivity
                self.rotation_y += dx * 0.3
                self.rotation_x += dy * 0.3
                
                # Normalize rotation angles
                self.rotation_x = self.rotation_x % 360
                self.rotation_y = self.rotation_y % 360
                
                # Update renderer view parameters
                self.renderer.update_view_params(
                    self.rotation_x, self.rotation_y, self.scale, self.translation_z, self.rotation_z
                )
                
                self.mouse_dragging = True
                #print(f"Rotating to X={self.rotation_x:.1f}, Y={self.rotation_y:.1f}")
            
        # Always update mouse position for next frame
        self.last_mouse_x = x
        self.last_mouse_y = y
    
    def mouse_button_callback(self, window, button, action, mods):
        """Handle mouse button events."""
        # Handle rotation if not over ImGui UI
        if not self.imgui_io.want_capture_mouse:
            if button == glfw.MOUSE_BUTTON_LEFT:
                if action == glfw.PRESS:
                    # Get current cursor position
                    x, y = glfw.get_cursor_pos(window)
                    
                    # Only start rotation if not over UI
                    if not self.is_point_in_ui(x, y):
                        self.mouse_pressed = True
                        self.mouse_dragging = False
                        self.last_mouse_x = x
                        self.last_mouse_y = y
                        #print(f"Mouse pressed at ({x}, {y})")
                elif action == glfw.RELEASE:
                    self.mouse_pressed = False
                    self.mouse_dragging = False
                    #print("Mouse released")
    
    def scroll_callback(self, window, xoffset, yoffset):
        """Handle mouse scroll for zooming."""
        # Direct handling of zoom, ignoring ImGui's scroll handling
        # This ensures our zoom always works regardless of ImGui state
        zoom_factor = 1.1
        
        # Get mouse position to check if it's over UI
        x, y = glfw.get_cursor_pos(window)
        is_over_ui = self.is_point_in_ui(x, y)
        
        # Only zoom if not over UI
        if not is_over_ui:
            if yoffset > 0:
                # Zoom in
                self.scale *= zoom_factor
                #print(f"Zoom callback in: {self.scale:.4f}")
            else:
                # Zoom out
                self.scale /= zoom_factor
                #print(f"Zoom callback out: {self.scale:.4f}")
            
            # Clamp scale
            self.scale = max(0.001, min(100.0, self.scale))
            
            # Update renderer
            self.renderer.update_view_params(
                self.rotation_x, self.rotation_y, self.scale, self.translation_z, self.rotation_z
            )
    
    def update_fps(self):
        """Update FPS counter."""
        current_time = time.time()
        self.frame_count += 1
        
        # Update FPS every second
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def render_ui(self):
        """Render ImGui UI."""
        imgui.new_frame()

        # Show Earth Civilization Status in upper right corner when active
        if self.simulation.is_earth_civilization_active():
            # Calculate position for upper right corner, leaving some margin
            text = "Earth Civilization Radio Active"
            text_width = imgui.calc_text_size(text).x * 2.2
            text_height = imgui.get_text_line_height() * 2.2
            window_width = imgui.get_window_width()
            
            # Set window position and size for the label
            imgui.set_next_window_position(self.width - text_width, 10)
            imgui.set_next_window_size(text_width + 20, text_height + 20)
            imgui.begin("Earth Status", False, 
                       imgui.WINDOW_NO_TITLE_BAR | 
                       imgui.WINDOW_NO_RESIZE | 
                       imgui.WINDOW_NO_MOVE |
                       imgui.WINDOW_NO_SCROLLBAR |
                       imgui.WINDOW_NO_COLLAPSE)
            
            # Set text color to bright green and display the status
            imgui.push_style_color(imgui.COLOR_TEXT, 0.0, 1.0, 0.0, 1.0)
            imgui.push_font(self.label_font)
            imgui.text(text)
            imgui.pop_font()
            imgui.pop_style_color()
            imgui.end()
        
        # Parameters Window
        # Position window on the left side of the screen
        imgui.set_next_window_position(10, 10, imgui.ALWAYS)
        # Make window height match screen height minus margins
        imgui.set_next_window_size(512, self.height - 20, imgui.ALWAYS)
        
        imgui.begin("Settings Panel", True, imgui.WINDOW_NO_MOVE)
        
        # ... rest of the existing UI code ...
        changed = False
        
        # Galaxy Parameters
        imgui.text("Galaxy Parameters")
        imgui.separator()
        
        imgui.set_next_item_width(150)
        changed, self.text_buffers["galaxy_diameter"] = imgui.input_text(
            "Galaxy Diameter (ly)", self.text_buffers["galaxy_diameter"], 16
        )
        imgui.set_next_item_width(150)
        changed, self.text_buffers["num_stars"] = imgui.input_text(
            "Number of Stars", self.text_buffers["num_stars"], 16
        )
        imgui.set_next_item_width(150)
        changed, self.text_buffers["num_radio_emitting"] = imgui.input_text(
            "Radio-Emitting Stars", self.text_buffers["num_radio_emitting"], 16
        )
        
        imgui.spacing()
        
        # Simulation Parameters
        imgui.text("Simulation Parameters")
        imgui.separator()
        imgui.set_next_item_width(150)
        changed, self.text_buffers["simulation_speed"] = imgui.input_text(
            "Simulation Speed (x)", self.text_buffers["simulation_speed"], 16
        )
        imgui.set_next_item_width(150)
        changed, self.text_buffers["simulation_duration"] = imgui.input_text(
            "Simulation Duration (y)", self.text_buffers["simulation_duration"], 16
        )
        
        imgui.spacing()
        
        # Radio Parameters
        imgui.text("Alien and Earth Civilizations")
        imgui.separator()
        imgui.set_next_item_width(150)
        changed, self.text_buffers["radio_signal_duration"] = imgui.input_text(
            "Alien Radio-able Civ Lifetime(y)", self.text_buffers["radio_signal_duration"], 16
        )
        imgui.set_next_item_width(150)
        changed, self.text_buffers["radio_blast_duration"] = imgui.input_text(
            "Radio Blast Interval (y)", self.text_buffers["radio_blast_duration"], 16
        )
        imgui.set_next_item_width(150)
        changed, self.text_buffers["radio_civilization_lifetime"] = imgui.input_text(
            "Earth Radio-able Civ Lifetime(y)", self.text_buffers["radio_civilization_lifetime"], 16
        )
        imgui.set_next_item_width(150)
        changed, self.text_buffers["sky_coverage_percent"] = imgui.input_text(
            "Earth Sky Coverage (%)", self.text_buffers["sky_coverage_percent"], 16
        )
        
        imgui.spacing()
        imgui.separator()
        
        # Control Buttons
        if self.is_running:
            if imgui.button("Pause", width=200):
                self.is_running = False
                self.simulation.pause()
        else:
            if imgui.button("Start/Resume", width=200):
                self.is_running = True
                self.update_simulation_params()
                self.simulation.start()
        
        imgui.same_line()
        if imgui.button("Reset", width=200):
            self.is_running = False
            self.update_simulation_params()
        
        imgui.spacing()
        imgui.separator()
        
        # Results Display
        imgui.text("Simulation Results")
        
        summary = self.simulation.get_summary()
        total_hits = summary['total_hits']
        current_time = summary['total_time']
        total_bursts = summary['total_bursts_emitted']
        
        imgui.push_style_color(imgui.COLOR_TEXT, 0.0, 1.0, 0.0, 1.0)
        imgui.text(f"Detection: {total_hits}")
        imgui.pop_style_color()
        imgui.text(f"Time: {current_time:,.0f} years")
        imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 1.0, 0.0, 1.0)
        imgui.text(f"Total Bursts Emitted: {total_bursts}")
        imgui.pop_style_color()
        
        imgui.spacing()
        
        # FPS Counter
        imgui.text(f"FPS: {self.fps}")
        
        # View control instructions
        imgui.text("Controls:")
        imgui.text("  • Left-click & drag to rotate")
        imgui.text("  • Scroll wheel to zoom")
        
        # View parameters display
        imgui.text(f"Rotation: X={self.rotation_x:.1f}, Y={self.rotation_y:.1f}, Z={self.rotation_z:.1f}")
        imgui.text(f"Zoom: {self.scale:.3f}")
        
        # Get current window position and size for UI interaction
        window_pos = imgui.get_window_position()
        window_size = imgui.get_window_size()
        self.ui_rect = (window_pos.x, window_pos.y, window_size.x, window_size.y)
        
        imgui.end()
        
        imgui.render()
        self.imgui_renderer.render(imgui.get_draw_data())
    
    def update_simulation_params(self):
        """Update simulation parameters from UI inputs."""
        try:
            galaxy_diameter = float(self.text_buffers["galaxy_diameter"])
            num_stars = int(self.text_buffers["num_stars"])
            num_radio_emitting = int(self.text_buffers["num_radio_emitting"])
            simulation_speed = float(self.text_buffers["simulation_speed"])
            simulation_duration = float(self.text_buffers["simulation_duration"])
            radio_signal_duration = float(self.text_buffers["radio_signal_duration"])
            radio_blast_duration = float(self.text_buffers["radio_blast_duration"])
            radio_civilization_lifetime = float(self.text_buffers["radio_civilization_lifetime"])
            sky_coverage_percent = float(self.text_buffers["sky_coverage_percent"])
            
            # Create a new simulation with updated parameters
            self.simulation = Simulation(
                galaxy_diameter=galaxy_diameter,
                num_stars=num_stars,
                num_radio_emitting=num_radio_emitting,
                simulation_speed=simulation_speed,
                simulation_duration=simulation_duration,
                radio_signal_duration=radio_signal_duration,
                radio_civilization_lifetime=radio_civilization_lifetime,
                radio_blast_duration=radio_blast_duration,
                sky_coverage_percent=sky_coverage_percent
            )
            
            # Update renderer with new star data
            if self.renderer:
                self.renderer.update_stars_data(self.simulation.get_stars_data())
                # Reset renderer's internal state to properly handle the new star data
                self.renderer.static_data_initialized = False
            
            return True
        except ValueError as e:
            print(f"Error: Invalid input - {str(e)}")
            return False
    
    def update(self):
        """Update the simulation."""
        # Calculate real time elapsed since last update
        current_time = time.time()
        elapsed = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Apply auto-rotation if enabled and not being manually rotated
        if self.auto_rotation_enabled and not self.mouse_pressed:
            # Calculate the amount of rotation based on time elapsed
            auto_rotation_amount = elapsed * self.auto_rotation_speed
            
            # Apply rotation to Z-axis
            self.rotation_z = (self.rotation_z - auto_rotation_amount) % 360
            
            # Update renderer view parameters with the new rotation
            self.renderer.update_view_params(
                self.rotation_x, self.rotation_y, self.scale, self.translation_z, self.rotation_z
            )
        
        # Update the simulation (using years as the time unit)
        if self.is_running:
            self.simulation.update(elapsed)
            
            # Check if simulation is complete
            if not self.simulation.is_running:
                self.is_running = False
        
        # WE DONT NEED TO UPDATE ALL THE STATIC STARS ON EACH FRAME
        # TODO We only need to update earth's star in an efficient way
        # Update renderer with simulation data
        #self.renderer.update_stars_data(self.simulation.get_stars_data())
        
        self.renderer.update_waves_data(self.simulation.get_active_waves())
        
        # Update debug info
        burst_count = self.simulation.total_bursts_emitted if hasattr(self.simulation, 'total_bursts_emitted') else 0
        wave_count = len(self.simulation.get_active_waves())
        self.renderer.update_debug_info(wave_count, burst_count)
    
    def handle_inputs(self):
        """
        Handle mouse inputs directly for more control.
        This adds additional mouse handling beyond the callbacks.
        """
        # Direct left mouse button check
        if glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            x, y = glfw.get_cursor_pos(self.window)
            
            # Only handle if not over ImGui UI
            if not self.imgui_io.want_capture_mouse and not self.is_point_in_ui(x, y):
                if not self.mouse_pressed:
                    # Initial press
                    self.mouse_pressed = True
                    self.last_mouse_x = x
                    self.last_mouse_y = y
                else:
                    # Continue dragging
                    dx = x - self.last_mouse_x
                    dy = y - self.last_mouse_y
                    
                    if dx != 0 or dy != 0:  # Only rotate if mouse actually moved
                        # Apply rotation
                        self.rotation_y += dx * 0.3
                        self.rotation_x += dy * 0.3
                        
                        # Normalize angles
                        self.rotation_x %= 360
                        self.rotation_y %= 360
                        
                        # Update renderer view
                        self.renderer.update_view_params(
                            self.rotation_x, self.rotation_y, self.scale, self.translation_z, self.rotation_z
                        )
                        
                    # Update position for next frame
                    self.last_mouse_x = x
                    self.last_mouse_y = y
        else:
            # Mouse released
            self.mouse_pressed = False
        
        # Manual detection of scroll wheel
        # This is a more direct approach to handle scroll wheel input
        # Some GLFW implementations might not properly forward scroll events to callbacks
        # in certain configurations, so this is a backup approach
    
    def run(self):
        """Run the main application loop."""
        # Initialize GLFW and OpenGL
        if not self.init_glfw():
            return
        
        # Load the stars (and earth) initially
        self.renderer.update_stars_data(self.simulation.get_stars_data())
        
        # Main loop
        while not glfw.window_should_close(self.window):
            # Poll events
            glfw.poll_events()
            
            # Handle direct input before ImGui
            self.handle_inputs()
            
            # Process ImGui inputs but ignore its scroll handling
            # This is important because we want our own scroll handling to take precedence
            self.imgui_renderer.process_inputs()
            
            # Update simulation
            self.update()
            
            # Render scene
            self.renderer.draw()
            
            # Render UI
            self.render_ui()
            
            # Swap buffers
            glfw.swap_buffers(self.window)
            
            # Update FPS counter
            self.update_fps()
        
        # Cleanup
        self.imgui_renderer.shutdown()
        self.renderer.cleanup()
        glfw.terminate()


if __name__ == "__main__":
    app = RadioWaveSimApp()
    app.run()