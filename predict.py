# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import os
from cog import BasePredictor, Input, Path
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = MusicGen.get_pretrained('large')

    def predict(
        self,
        description: str = Input(description="Music description for generation"),
        duration: int = Input(description="Duration of the generated audio in seconds", default=8)
    ) -> Path:
        """Run a single prediction on the model"""
        self.model.set_generation_params(duration)  # generate 8 seconds.
        wav = self.model.generate([description])
        output_path = os.path.join("/tmp", f"output.wav")
        audio_write(output_path, wav[0].cpu(), self.model.sample_rate, strategy="loudness")
        return Path(output_path)
