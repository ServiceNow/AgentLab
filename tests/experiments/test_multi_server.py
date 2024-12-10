from agentlab.experiments.multi_server import WebArenaInstanceVars
from browsergym.webarena.instance import WebArenaInstance


def test_webarena_multiserver():
    instance = WebArenaInstanceVars.from_env_vars()
    instance_1 = instance.clone()
    instance_1.base_url = "http://webarena1.eastus.cloudapp.azure.com"
    instance_1.init()

    bgym_instance = WebArenaInstance()
    base_url_1 = bgym_instance.urls["reddit"].rsplit(":", 1)[0]
    assert base_url_1 == instance_1.base_url

    instance_2 = instance.clone()
    instance_2.base_url = "http://webarena2.eastus.cloudapp.azure.com"
    instance_2.init()

    bgym_instance = WebArenaInstance()
    base_url_2 = bgym_instance.urls["reddit"].rsplit(":", 1)[0]
    assert base_url_2 == instance_2.base_url


if __name__ == "__main__":
    test_webarena_multiserver()
