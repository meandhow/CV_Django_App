from django import forms
from uploadimg.models import Image
class ImageForm(forms.ModelForm):
    """Form for the image model"""
    class Meta:
        model = Image
        fields = ('image','detector_model')

