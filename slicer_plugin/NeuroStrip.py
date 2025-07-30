import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

import slicer
import vtk
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
)
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

try:
    import neurostrip
except ImportError:
    slicer.util.pip_install("neurostrip[cpu]")
    import neurostrip


#
# NeuroStrip
#


class NeuroStrip(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("NeuroStrip")
        self.parent.categories = [
            translate("qSlicerAbstractCoreModule", "Segmentation")
        ]
        self.parent.dependencies = ["Segmentations"]
        self.parent.contributors = ["Bryn Lloyd (Slicer Community)"]
        self.parent.helpText = _("""
This module provides CNN-based skull stripping (brain masking) from MRI using the NeuroStrip library.
It can generate brain masks and apply them to create masked brain images.
See more information in the <a href="https://github.com/dyollb/neurostrip">module documentation</a>.
""")
        self.parent.acknowledgementText = _("""
This file was originally developed by Bryn Lloyd.
The NeuroStrip algorithm is based on deep learning techniques for medical image segmentation.
""")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly),
    # it is recommended to store data sets in a separate repository and reference them here by URL.
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="NeuroStrip",
        sampleName="NeuroStrip1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "NeuroStrip1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="NeuroStrip.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="NeuroStrip1",
    )


#
# NeuroStripParameterNode
#


@parameterNodeWrapper
class NeuroStripParameterNode:
    """
    The parameters needed by module.
    """

    inputVolume: slicer.vtkMRMLScalarVolumeNode
    outputMask: slicer.vtkMRMLLabelMapVolumeNode
    outputMaskedVolume: slicer.vtkMRMLScalarVolumeNode


#
# NeuroStripWidget
#


class NeuroStripWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/NeuroStrip.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Explicitly set the MRML scene for the node combo boxes
        self.ui.inputVolume.setMRMLScene(slicer.mrmlScene)
        self.ui.outputMask.setMRMLScene(slicer.mrmlScene)
        self.ui.outputMaskedVolume.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = NeuroStripLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(
            slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose
        )
        self.addObserver(
            slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose
        )

        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(
                self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply
            )

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass(
                "vtkMRMLScalarVolumeNode"
            )
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode

    def setParameterNode(
        self, inputParameterNode: Optional[NeuroStripParameterNode]
    ) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(
                self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply
            )
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(
                self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply
            )
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        if (
            self._parameterNode
            and self._parameterNode.inputVolume
            and self._parameterNode.outputMask
        ):
            self.ui.applyButton.toolTip = _(
                "Compute brain mask and optionally masked volume"
            )
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select input volume and output mask")
            self.ui.applyButton.enabled = False

    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(
            _("Failed to compute results."), waitCursor=True
        ):
            # Compute output
            self.logic.process(
                self._parameterNode.inputVolume,
                self._parameterNode.outputMask,
                self._parameterNode.outputMaskedVolume,
            )


#
# NeuroStripLogic
#


class NeuroStripLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by this module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return NeuroStripParameterNode(super().getParameterNode())

    def process(
        self,
        inputVolume: slicer.vtkMRMLScalarVolumeNode,
        outputMask: slicer.vtkMRMLLabelMapVolumeNode,
        outputMaskedVolume: slicer.vtkMRMLScalarVolumeNode = None,
    ) -> None:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be processed
        :param outputMask: brain mask result
        :param outputMaskedVolume: masked brain volume result (optional)
        """

        if not inputVolume or not outputMask:
            raise ValueError("Input volume and output mask are required")

        logging.info("Processing started")

        # Create temporary files for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)

            # Save input volume to temporary file
            input_path = temp_dir_path / "input.nii.gz"
            mask_path = temp_dir_path / "mask.nii.gz"
            masked_path = (
                temp_dir_path / "masked.nii.gz" if outputMaskedVolume else None
            )

            # Export input volume
            slicer.util.exportNode(inputVolume, str(input_path))

            # Run NeuroStrip prediction
            try:
                neurostrip.lib.predict(
                    image_path=input_path,
                    mask_path=mask_path,
                    masked_image_path=masked_path,
                    device="cpu",
                )
            except Exception as e:
                logging.error(f"NeuroStrip prediction failed: {e}")
                raise e

            # Load results back into Slicer
            if mask_path.exists():
                # Load mask without specifying name to avoid conflicts
                tempMaskNode = slicer.util.loadLabelVolume(str(mask_path))
                # Copy the loaded volume to the output node
                if tempMaskNode:
                    outputMask.SetAndObserveImageData(tempMaskNode.GetImageData())
                    outputMask.SetOrigin(tempMaskNode.GetOrigin())
                    outputMask.SetSpacing(tempMaskNode.GetSpacing())
                    directionMatrix = np.eye(3, 3).tolist()
                    tempMaskNode.GetIJKToRASDirections(directionMatrix)
                    outputMask.SetIJKToRASDirections(directionMatrix)
                    slicer.mrmlScene.RemoveNode(tempMaskNode)

            if outputMaskedVolume and masked_path and masked_path.exists():
                # Load masked volume without specifying name to avoid conflicts
                tempMaskedNode = slicer.util.loadVolume(str(masked_path))
                # Copy the loaded volume to the output node
                if tempMaskedNode:
                    outputMaskedVolume.SetAndObserveImageData(
                        tempMaskedNode.GetImageData()
                    )
                    outputMaskedVolume.SetOrigin(tempMaskedNode.GetOrigin())
                    outputMaskedVolume.SetSpacing(tempMaskedNode.GetSpacing())
                    directionMatrix = np.eye(3, 3).tolist()
                    tempMaskedNode.GetIJKToRASDirections(directionMatrix)
                    outputMaskedVolume.SetIJKToRASDirections(directionMatrix)

                    slicer.mrmlScene.RemoveNode(tempMaskedNode)

        logging.info("Processing completed")


#
# NeuroStripTest
#


class NeuroStripTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_neurostrip()

    def test_neurostrip(self):
        self.delayDisplay("Starting the test")

        # Get/create input data
        import SampleData

        registerSampleData()
        inputVolume = SampleData.downloadSample("NeuroStrip1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()

        # Test the module logic
        logic = NeuroStripLogic()

        # Create output nodes
        outputMask = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        outputMaskedVolume = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLScalarVolumeNode"
        )

        # Test algorithm with CPU device
        logic.process(inputVolume, outputMask, outputMaskedVolume)

        # Verify results
        self.assertIsNotNone(outputMask.GetImageData())
        self.assertIsNotNone(outputMaskedVolume.GetImageData())

        maskScalarRange = outputMask.GetImageData().GetScalarRange()
        self.assertEqual(maskScalarRange[0], 0)
        self.assertEqual(maskScalarRange[1], 1)

        outputScalarRange = outputMaskedVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], 0)
        self.assertEqual(outputScalarRange[1], 605)

        self.delayDisplay("Test passed")
