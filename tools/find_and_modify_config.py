import json
import os
import tempfile


def find_config_file():
    # Check common locations used by OpenBCI_GUI across platforms.
    # Paths using '~' are expanded to the current user's home directory.
    config_paths = [
        "./config.json",
        "~/Documents/OpenBCI_GUI/Settings/config.json",
        "~/Library/Application Support/OpenBCI/config.json",
        "~/.config/OpenBCI/config.json",
        "~/AppData/Local/OpenBCI/config.json",
    ]

    for path in config_paths:
        expanded_path = os.path.expanduser(path)
        if os.path.exists(expanded_path):
            return expanded_path
    return None


def modify_settings_to_text():
    config_file = find_config_file()
    if config_file:
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                settings = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: could not parse JSON in {config_file}: {e}")
            return

        # Modify settings for text output
        settings['display_format'] = 'text'
        settings['output_mode'] = 'text'
        settings['visual_mode'] = False

        # Write to a temporary file first to avoid corrupting the original
        # if an error occurs during the write.
        config_dir = os.path.dirname(os.path.abspath(config_file))
        try:
            fd, tmp_path = tempfile.mkstemp(dir=config_dir, suffix='.json')
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as tmp_file:
                    json.dump(settings, tmp_file, indent=2)
                os.replace(tmp_path, config_file)
            except OSError:
                os.unlink(tmp_path)
                raise
        except OSError as e:
            print(f"Error: could not write settings to {config_file}: {e}")
            return

        print(f"Settings updated in {config_file}")
    else:
        print("No configuration file found")


if __name__ == "__main__":
    modify_settings_to_text()
