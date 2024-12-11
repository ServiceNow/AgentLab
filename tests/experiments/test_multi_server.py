from agentlab.experiments.multi_server import WebArenaInstanceVars
from browsergym.webarena.instance import WebArenaInstance


def test_webarena_multiserver():

    instance_1 = WebArenaInstanceVars(
        base_url="http://webarena1.eastus.cloudapp.azure.com",
        shopping="8082/",
        shopping_admin="8083/admin",
        reddit="8080",
        gitlab="9001",
        wikipedia="8081/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing",
        map="443",
        homepage="80",
        full_reset="7565",
        module_name="webarena",
        prefix="WA_",
    )

    instance_1.init()

    bgym_instance = WebArenaInstance()
    base_url_1 = bgym_instance.urls["reddit"].rsplit(":", 1)[0]
    assert base_url_1 == instance_1.base_url

    instance_2 = instance_1.clone()
    instance_2.base_url = "http://webarena2.eastus.cloudapp.azure.com"
    instance_2.init()

    bgym_instance = WebArenaInstance()
    base_url_2 = bgym_instance.urls["reddit"].rsplit(":", 1)[0]
    assert base_url_2 == instance_2.base_url


if __name__ == "__main__":
    test_webarena_multiserver()
