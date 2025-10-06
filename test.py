# create a isaacsim window
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
# app_launcher.launch()
from pxr import Usd, UsdGeom, UsdPhysics, Gf, PhysxSchema
import omni.usd

# Create a stage
omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()

# Define the root Xform (transformable object)
rootxform = UsdGeom.Xform.Define(stage, "/World")

rigidBodyPaths = ["/World/rigidBody0", "/World/rigidBody1"]
revoluteJointPath = "/World/revoluteJoint"
fixedJointPath = "/World/fixedJoint"

# The initial pose of the root link.
rootLinkStartPosition = Gf.Vec3f(0, 0, 0)
rootLinkStartRotation = Gf.Quatf(1.0)
# The joint frames of the revolute joint coupling
# the two links of the articulation.
revoluteJointlocalPositions = [Gf.Vec3f(0.0, 10.0, 0.0), Gf.Vec3f(0.0, 0.0, 0.0)]
revoluteJointLocalRotations = [Gf.Quatf(1.0), Gf.Quatf(1.0)]

# body0 is chosen to be the root link.
rootLinkId = 0
rigidBodyXforms = [None] * 2

for i in range(2):
    # Create the rigid body
    rigidBodyXform = UsdGeom.Xform.Define(stage, rigidBodyPaths[i])
    rigidBodyXforms[i] = rigidBodyXform
    rigidBodyPrim = rigidBodyXform.GetPrim()
    rigidBodyAPI = UsdPhysics.RigidBodyAPI.Apply(rigidBodyPrim)
    rigidBodyAPI.CreateRigidBodyEnabledAttr(True)
    massAPI = UsdPhysics.MassAPI.Apply(rigidBodyPrim)
    massAPI.CreateMassAttr(2.0)


# Create a revolute joint between the two rigid body prims
revoluteJoint = UsdPhysics.RevoluteJoint.Define(stage, revoluteJointPath)

breakpoint()
revoluteJoint.CreateAxisAttr(UsdPhysics.Tokens.y)
revoluteJoint.CreateBody0Rel().AddTarget(rigidBodyPaths[0])
revoluteJoint.CreateBody1Rel().AddTarget(rigidBodyPaths[1])
revoluteJoint.CreateLocalPos0Attr().Set(revoluteJointlocalPositions[0])
revoluteJoint.CreateLocalRot0Attr().Set(revoluteJointLocalRotations[0])
revoluteJoint.CreateLocalPos1Attr().Set(revoluteJointlocalPositions[1])
revoluteJoint.CreateLocalRot1Attr().Set(revoluteJointLocalRotations[1])

# Create a fixed joint between the root link and the world.
# Mark the fixed joint as the root. This will create a fixed
# base articulation with body0 as the root link.
fixedJoint = UsdPhysics.FixedJoint.Define(stage, fixedJointPath)
fixedJoint.CreateBody0Rel().AddTarget(rigidBodyPaths[rootLinkId])
UsdPhysics.ArticulationRootAPI.Apply(fixedJoint.GetPrim())

# Set the initial pose of the root link
rigidBodyXforms[rootLinkId].AddTranslateOp().Set(rootLinkStartPosition)
rigidBodyXforms[rootLinkId].AddOrientOp().Set(rootLinkStartRotation)

from loguru import logger as log
log.info("Done")