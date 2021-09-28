from safe_il.envs.simple_navigation import SimpleNavigation
from safe_il.envs.bone_drilling_2d import BoneDrilling2D
from safe_il.envs.cartpole import CartPole

ENVS = {
    "navigation": SimpleNavigation,
    "drilling": BoneDrilling2D,
    "cartpole": CartPole,
}