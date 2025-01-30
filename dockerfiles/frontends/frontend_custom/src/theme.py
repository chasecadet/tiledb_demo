from typing import Union
from gradio.themes import Base
from gradio.themes.utils import colors

# Define TileDB colors here
TILEDB_PRIMARY = "##2a7de1"  # Example primary color
TILEDB_SECONDARY = "#F3F4F6"  # Example secondary color
TILEDB_BACKGROUND = "#374151"  # Example background color
TILEDB_TEXT = "#6E707A"  # Example text color

BORDER_WIDTH = "3px"
BORDER_RADIUS = "100px"
INPUT_BORDER_WIDTH = "1px"

class TileDBTheme(Base):
    def __init__(
            self,
            *,
            primary_hue: Union[colors.Color, str] = colors.slate,
            neutral_hue: Union[colors.Color, str] = colors.cyan,
            ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=primary_hue,
            neutral_hue=neutral_hue
        )
        
        super().set(
            button_border_width=BORDER_WIDTH,
            button_border_width_dark=BORDER_WIDTH,
            button_primary_background_fill=TILEDB_PRIMARY,
            button_primary_background_fill_dark=TILEDB_BACKGROUND,
            button_primary_text_color=TILEDB_SECONDARY,
            button_primary_border_color=TILEDB_PRIMARY,
            button_primary_border_color_dark=TILEDB_PRIMARY,
            button_primary_border_color_hover=TILEDB_PRIMARY,
            button_primary_border_color_hover_dark=TILEDB_PRIMARY,
            button_primary_background_fill_hover=TILEDB_PRIMARY,
            button_large_radius=BORDER_RADIUS,
            button_secondary_background_fill=TILEDB_SECONDARY,
            button_secondary_background_fill_dark=TILEDB_BACKGROUND,
            button_secondary_border_color_hover=TILEDB_PRIMARY,
            button_secondary_border_color_hover_dark=TILEDB_PRIMARY,
            button_secondary_border_color=TILEDB_PRIMARY,
            button_secondary_border_color_dark=TILEDB_BACKGROUND,
            button_secondary_text_color=TILEDB_TEXT,
            slider_color=TILEDB_PRIMARY,
            checkbox_border_color_hover=TILEDB_PRIMARY,
            checkbox_border_color_hover_dark=TILEDB_PRIMARY,
            input_border_width=INPUT_BORDER_WIDTH,
            input_border_width_dark=INPUT_BORDER_WIDTH,
            input_border_color_focus=TILEDB_PRIMARY,
            input_border_color_focus_dark=TILEDB_PRIMARY,
            input_border_color_hover=TILEDB_PRIMARY,
            input_border_color_hover_dark=TILEDB_PRIMARY
            )

