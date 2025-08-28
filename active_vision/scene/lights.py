from metasim.scenario.lights import DiskLightCfg, DistantLightCfg, DomeLightCfg

lights = [
    # Sky dome light - provides soft ambient lighting from all directions
    DomeLightCfg(
        intensity=800.0,  # Moderate ambient lighting
        color=(0.85, 0.9, 1.0),  # Slightly blue sky color
    ),
    # Sun light - main directional light
    DistantLightCfg(
        intensity=1200.0,  # Strong sunlight
        polar=35.0,  # Sun at 35° elevation (natural angle)
        azimuth=60.0,  # From the northeast
        color=(1.0, 0.98, 0.95),  # Slightly warm sunlight
    ),
    # Soft area light for subtle fill
    DiskLightCfg(
        intensity=300.0,
        radius=1.5,  # Large disk for soft light
        pos=(2.0, -2.0, 4.0),  # Side fill light
        rot=(0.7071, 0.7071, 0.0, 0.0),  # Angled towards scene
        color=(0.95, 0.95, 1.0),
    ),
]
