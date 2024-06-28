from pathlib import Path
import pandas as pd
from agentlab.analyze import inspect_results
from agentlab.experiments.task_collections import webgum_tasks
from browsergym.experiments.loop import ExpResult, yield_all_exp_results

from browsergym.core.action.parsers import highlevel_action_parser
from tqdm import tqdm
from joblib import Memory


def select_single_agent(result_df: pd.DataFrame, agent_index) -> pd.DataFrame:
    """
    Selects the rows of the dataframe that correspond to a single agent
    """
    new_df = result_df.reset_index(level="task_name", inplace=False).sort_index()
    agent_result_df = new_df.loc[agent_index]

    inspect_results.set_index_from_variables(agent_result_df)
    return agent_result_df


NAME_MAP = {
    "openai/gpt-4-1106-preview": "gpt-4",
    "openai/gpt-4-vision-preview": "gpt-4-v",
    "openai/gpt-3.5-turbo-1106": "gpt-3.5",
    "agent_gpt-3.5": "gpt-3.5",
    "agent_gpt-4o": "gpt-4o",
    "agent_gpt-4o_vision": "gpt-4o-v",
    "agent_llama3": "Llama3",
    "GenericAgent": "Llama3",
    "miniwob": "MiniWoB++",
    "workarna": "WorkArena",
    "agent_gpt-4o_L3": "gpt-4o",
    "agent_gpt-4o_L3_vision": "gpt-4o-v",
    "webarena.0": "information-seeking",
    "webarena.1": "information-seeking",
    "webarena.2": "information-seeking",
    "webarena.3": "information-seeking",
    "webarena.4": "information-seeking",
    "webarena.5": "information-seeking",
    "webarena.6": "information-seeking",
    "webarena.7": "information-seeking",
    "webarena.8": "information-seeking",
    "webarena.9": "information-seeking",
    "webarena.10": "information-seeking",
    "webarena.11": "information-seeking",
    "webarena.12": "information-seeking",
    "webarena.13": "information-seeking",
    "webarena.14": "information-seeking",
    "webarena.15": "information-seeking",
    "webarena.16": "information-seeking",
    "webarena.17": "information-seeking",
    "webarena.18": "information-seeking",
    "webarena.19": "information-seeking",
    "webarena.20": "information-seeking",
    "webarena.21": "information-seeking",
    "webarena.22": "information-seeking",
    "webarena.23": "information-seeking",
    "webarena.24": "information-seeking",
    "webarena.25": "information-seeking",
    "webarena.26": "information-seeking",
    "webarena.27": "information-seeking",
    "webarena.28": "information-seeking",
    "webarena.29": "information-seeking",
    "webarena.30": "information-seeking",
    "webarena.31": "information-seeking",
    "webarena.32": "information-seeking",
    "webarena.33": "information-seeking",
    "webarena.34": "information-seeking",
    "webarena.35": "information-seeking",
    "webarena.36": "information-seeking",
    "webarena.37": "information-seeking",
    "webarena.38": "information-seeking",
    "webarena.39": "information-seeking",
    "webarena.40": "information-seeking",
    "webarena.41": "information-seeking",
    "webarena.42": "information-seeking",
    "webarena.43": "information-seeking",
    "webarena.44": "navigation",
    "webarena.45": "navigation",
    "webarena.46": "navigation",
    "webarena.47": "information-seeking",
    "webarena.48": "information-seeking",
    "webarena.49": "information-seeking",
    "webarena.50": "information-seeking",
    "webarena.51": "information-seeking",
    "webarena.52": "information-seeking",
    "webarena.53": "information-seeking",
    "webarena.54": "information-seeking",
    "webarena.55": "information-seeking",
    "webarena.56": "information-seeking",
    "webarena.57": "information-seeking",
    "webarena.58": "information-seeking",
    "webarena.59": "information-seeking",
    "webarena.60": "information-seeking",
    "webarena.61": "information-seeking",
    "webarena.62": "information-seeking",
    "webarena.63": "information-seeking",
    "webarena.64": "information-seeking",
    "webarena.65": "information-seeking",
    "webarena.66": "information-seeking",
    "webarena.67": "information-seeking",
    "webarena.68": "information-seeking",
    "webarena.69": "information-seeking",
    "webarena.70": "information-seeking",
    "webarena.71": "information-seeking",
    "webarena.72": "information-seeking",
    "webarena.73": "information-seeking",
    "webarena.74": "information-seeking",
    "webarena.75": "information-seeking",
    "webarena.76": "information-seeking",
    "webarena.77": "information-seeking",
    "webarena.78": "information-seeking",
    "webarena.79": "information-seeking",
    "webarena.80": "information-seeking",
    "webarena.81": "information-seeking",
    "webarena.82": "information-seeking",
    "webarena.83": "information-seeking",
    "webarena.84": "information-seeking",
    "webarena.85": "information-seeking",
    "webarena.86": "information-seeking",
    "webarena.87": "information-seeking",
    "webarena.88": "information-seeking",
    "webarena.89": "information-seeking",
    "webarena.90": "information-seeking",
    "webarena.91": "information-seeking",
    "webarena.92": "information-seeking",
    "webarena.93": "information-seeking",
    "webarena.94": "information-seeking",
    "webarena.95": "information-seeking",
    "webarena.96": "information-seeking",
    "webarena.97": "information-seeking",
    "webarena.98": "information-seeking",
    "webarena.99": "information-seeking",
    "webarena.100": "information-seeking",
    "webarena.101": "information-seeking",
    "webarena.102": "navigation",
    "webarena.103": "navigation",
    "webarena.104": "navigation",
    "webarena.105": "navigation",
    "webarena.106": "navigation",
    "webarena.107": "information-seeking",
    "webarena.108": "information-seeking",
    "webarena.109": "information-seeking",
    "webarena.110": "information-seeking",
    "webarena.111": "information-seeking",
    "webarena.112": "information-seeking",
    "webarena.113": "information-seeking",
    "webarena.114": "information-seeking",
    "webarena.115": "information-seeking",
    "webarena.116": "information-seeking",
    "webarena.117": "information-seeking",
    "webarena.118": "content-and-config",
    "webarena.119": "information-seeking",
    "webarena.120": "information-seeking",
    "webarena.121": "information-seeking",
    "webarena.122": "information-seeking",
    "webarena.123": "information-seeking",
    "webarena.124": "information-seeking",
    "webarena.125": "information-seeking",
    "webarena.126": "information-seeking",
    "webarena.127": "information-seeking",
    "webarena.128": "information-seeking",
    "webarena.129": "information-seeking",
    "webarena.130": "information-seeking",
    "webarena.131": "information-seeking",
    "webarena.132": "information-seeking",
    "webarena.133": "information-seeking",
    "webarena.134": "information-seeking",
    "webarena.135": "information-seeking",
    "webarena.136": "information-seeking",
    "webarena.137": "information-seeking",
    "webarena.138": "information-seeking",
    "webarena.139": "information-seeking",
    "webarena.140": "information-seeking",
    "webarena.141": "information-seeking",
    "webarena.142": "information-seeking",
    "webarena.143": "information-seeking",
    "webarena.144": "information-seeking",
    "webarena.145": "information-seeking",
    "webarena.146": "information-seeking",
    "webarena.147": "information-seeking",
    "webarena.148": "information-seeking",
    "webarena.149": "information-seeking",
    "webarena.150": "information-seeking",
    "webarena.151": "information-seeking",
    "webarena.152": "information-seeking",
    "webarena.153": "information-seeking",
    "webarena.154": "information-seeking",
    "webarena.155": "information-seeking",
    "webarena.156": "navigation",
    "webarena.157": "navigation",
    "webarena.158": "navigation",
    "webarena.159": "navigation",
    "webarena.160": "navigation",
    "webarena.161": "navigation",
    "webarena.162": "navigation",
    "webarena.163": "information-seeking",
    "webarena.164": "information-seeking",
    "webarena.165": "information-seeking",
    "webarena.166": "information-seeking",
    "webarena.167": "information-seeking",
    "webarena.168": "information-seeking",
    "webarena.169": "information-seeking",
    "webarena.170": "information-seeking",
    "webarena.171": "information-seeking",
    "webarena.172": "information-seeking",
    "webarena.173": "navigation",
    "webarena.174": "navigation",
    "webarena.175": "navigation",
    "webarena.176": "navigation",
    "webarena.177": "navigation",
    "webarena.178": "navigation",
    "webarena.179": "navigation",
    "webarena.180": "navigation",
    "webarena.181": "navigation",
    "webarena.182": "navigation",
    "webarena.183": "information-seeking",
    "webarena.184": "information-seeking",
    "webarena.185": "information-seeking",
    "webarena.186": "information-seeking",
    "webarena.187": "information-seeking",
    "webarena.188": "information-seeking",
    "webarena.189": "information-seeking",
    "webarena.190": "information-seeking",
    "webarena.191": "information-seeking",
    "webarena.192": "information-seeking",
    "webarena.193": "information-seeking",
    "webarena.194": "information-seeking",
    "webarena.195": "information-seeking",
    "webarena.196": "information-seeking",
    "webarena.197": "information-seeking",
    "webarena.198": "information-seeking",
    "webarena.199": "information-seeking",
    "webarena.200": "information-seeking",
    "webarena.201": "information-seeking",
    "webarena.202": "information-seeking",
    "webarena.203": "information-seeking",
    "webarena.204": "information-seeking",
    "webarena.205": "information-seeking",
    "webarena.206": "information-seeking",
    "webarena.207": "information-seeking",
    "webarena.208": "information-seeking",
    "webarena.209": "information-seeking",
    "webarena.210": "information-seeking",
    "webarena.211": "information-seeking",
    "webarena.212": "information-seeking",
    "webarena.213": "information-seeking",
    "webarena.214": "information-seeking",
    "webarena.215": "information-seeking",
    "webarena.216": "information-seeking",
    "webarena.217": "information-seeking",
    "webarena.218": "information-seeking",
    "webarena.219": "information-seeking",
    "webarena.220": "information-seeking",
    "webarena.221": "information-seeking",
    "webarena.222": "information-seeking",
    "webarena.223": "information-seeking",
    "webarena.224": "information-seeking",
    "webarena.225": "information-seeking",
    "webarena.226": "information-seeking",
    "webarena.227": "information-seeking",
    "webarena.228": "information-seeking",
    "webarena.229": "information-seeking",
    "webarena.230": "information-seeking",
    "webarena.231": "information-seeking",
    "webarena.232": "information-seeking",
    "webarena.233": "information-seeking",
    "webarena.234": "information-seeking",
    "webarena.235": "information-seeking",
    "webarena.236": "information-seeking",
    "webarena.237": "information-seeking",
    "webarena.238": "navigation",
    "webarena.239": "navigation",
    "webarena.240": "navigation",
    "webarena.241": "navigation",
    "webarena.242": "navigation",
    "webarena.243": "information-seeking",
    "webarena.244": "information-seeking",
    "webarena.245": "information-seeking",
    "webarena.246": "information-seeking",
    "webarena.247": "information-seeking",
    "webarena.248": "information-seeking",
    "webarena.249": "information-seeking",
    "webarena.250": "information-seeking",
    "webarena.251": "information-seeking",
    "webarena.252": "information-seeking",
    "webarena.253": "information-seeking",
    "webarena.254": "information-seeking",
    "webarena.255": "information-seeking",
    "webarena.256": "information-seeking",
    "webarena.257": "information-seeking",
    "webarena.258": "navigation",
    "webarena.259": "information-seeking",
    "webarena.260": "navigation",
    "webarena.261": "navigation",
    "webarena.262": "navigation",
    "webarena.263": "navigation",
    "webarena.264": "navigation",
    "webarena.265": "information-seeking",
    "webarena.266": "information-seeking",
    "webarena.267": "information-seeking",
    "webarena.268": "information-seeking",
    "webarena.269": "navigation",
    "webarena.270": "navigation",
    "webarena.271": "navigation",
    "webarena.272": "navigation",
    "webarena.273": "navigation",
    "webarena.274": "navigation",
    "webarena.275": "navigation",
    "webarena.276": "navigation",
    "webarena.277": "navigation",
    "webarena.278": "navigation",
    "webarena.279": "information-seeking",
    "webarena.280": "information-seeking",
    "webarena.281": "information-seeking",
    "webarena.282": "information-seeking",
    "webarena.283": "navigation",
    "webarena.284": "navigation",
    "webarena.285": "navigation",
    "webarena.286": "navigation",
    "webarena.287": "information-seeking",
    "webarena.288": "information-seeking",
    "webarena.289": "information-seeking",
    "webarena.290": "information-seeking",
    "webarena.291": "information-seeking",
    "webarena.292": "information-seeking",
    "webarena.293": "information-seeking",
    "webarena.294": "information-seeking",
    "webarena.295": "information-seeking",
    "webarena.296": "information-seeking",
    "webarena.297": "information-seeking",
    "webarena.298": "navigation",
    "webarena.299": "navigation",
    "webarena.300": "navigation",
    "webarena.301": "information-seeking",
    "webarena.302": "information-seeking",
    "webarena.303": "information-seeking",
    "webarena.304": "information-seeking",
    "webarena.305": "information-seeking",
    "webarena.306": "information-seeking",
    "webarena.307": "information-seeking",
    "webarena.308": "information-seeking",
    "webarena.309": "information-seeking",
    "webarena.310": "information-seeking",
    "webarena.311": "information-seeking",
    "webarena.312": "information-seeking",
    "webarena.313": "information-seeking",
    "webarena.314": "information-seeking",
    "webarena.315": "information-seeking",
    "webarena.316": "information-seeking",
    "webarena.317": "information-seeking",
    "webarena.318": "information-seeking",
    "webarena.319": "information-seeking",
    "webarena.320": "information-seeking",
    "webarena.321": "information-seeking",
    "webarena.322": "information-seeking",
    "webarena.323": "information-seeking",
    "webarena.324": "navigation",
    "webarena.325": "navigation",
    "webarena.326": "navigation",
    "webarena.327": "navigation",
    "webarena.328": "navigation",
    "webarena.329": "information-seeking",
    "webarena.330": "information-seeking",
    "webarena.331": "information-seeking",
    "webarena.332": "information-seeking",
    "webarena.333": "information-seeking",
    "webarena.334": "information-seeking",
    "webarena.335": "information-seeking",
    "webarena.336": "information-seeking",
    "webarena.337": "information-seeking",
    "webarena.338": "information-seeking",
    "webarena.339": "navigation",
    "webarena.340": "navigation",
    "webarena.341": "navigation",
    "webarena.342": "navigation",
    "webarena.343": "navigation",
    "webarena.344": "information-seeking",
    "webarena.345": "information-seeking",
    "webarena.346": "information-seeking",
    "webarena.347": "information-seeking",
    "webarena.348": "information-seeking",
    "webarena.349": "information-seeking",
    "webarena.350": "information-seeking",
    "webarena.351": "navigation",
    "webarena.352": "navigation",
    "webarena.353": "navigation",
    "webarena.354": "navigation",
    "webarena.355": "navigation",
    "webarena.356": "content-and-config",
    "webarena.357": "navigation",
    "webarena.358": "information-seeking",
    "webarena.359": "information-seeking",
    "webarena.360": "information-seeking",
    "webarena.361": "information-seeking",
    "webarena.362": "information-seeking",
    "webarena.363": "information-seeking",
    "webarena.364": "information-seeking",
    "webarena.365": "information-seeking",
    "webarena.366": "information-seeking",
    "webarena.367": "information-seeking",
    "webarena.368": "information-seeking",
    "webarena.369": "content-and-config",
    "webarena.370": "content-and-config",
    "webarena.371": "content-and-config",
    "webarena.372": "content-and-config",
    "webarena.373": "content-and-config",
    "webarena.374": "navigation",
    "webarena.375": "navigation",
    "webarena.376": "information-seeking",
    "webarena.377": "navigation",
    "webarena.378": "navigation",
    "webarena.379": "navigation",
    "webarena.380": "navigation",
    "webarena.381": "navigation",
    "webarena.382": "information-seeking",
    "webarena.383": "information-seeking",
    "webarena.384": "information-seeking",
    "webarena.385": "information-seeking",
    "webarena.386": "information-seeking",
    "webarena.387": "information-seeking",
    "webarena.388": "information-seeking",
    "webarena.389": "content-and-config",
    "webarena.390": "content-and-config",
    "webarena.391": "content-and-config",
    "webarena.392": "content-and-config",
    "webarena.393": "content-and-config",
    "webarena.394": "content-and-config",
    "webarena.395": "content-and-config",
    "webarena.396": "content-and-config",
    "webarena.397": "content-and-config",
    "webarena.398": "content-and-config",
    "webarena.399": "content-and-config",
    "webarena.400": "content-and-config",
    "webarena.401": "content-and-config",
    "webarena.402": "content-and-config",
    "webarena.403": "content-and-config",
    "webarena.404": "content-and-config",
    "webarena.405": "content-and-config",
    "webarena.406": "content-and-config",
    "webarena.407": "content-and-config",
    "webarena.408": "content-and-config",
    "webarena.409": "content-and-config",
    "webarena.410": "content-and-config",
    "webarena.411": "content-and-config",
    "webarena.412": "content-and-config",
    "webarena.413": "content-and-config",
    "webarena.414": "content-and-config",
    "webarena.415": "content-and-config",
    "webarena.416": "content-and-config",
    "webarena.417": "content-and-config",
    "webarena.418": "content-and-config",
    "webarena.419": "content-and-config",
    "webarena.420": "content-and-config",
    "webarena.421": "content-and-config",
    "webarena.422": "content-and-config",
    "webarena.423": "content-and-config",
    "webarena.424": "content-and-config",
    "webarena.425": "content-and-config",
    "webarena.426": "content-and-config",
    "webarena.427": "content-and-config",
    "webarena.428": "content-and-config",
    "webarena.429": "content-and-config",
    "webarena.430": "content-and-config",
    "webarena.431": "content-and-config",
    "webarena.432": "content-and-config",
    "webarena.433": "content-and-config",
    "webarena.434": "content-and-config",
    "webarena.435": "content-and-config",
    "webarena.436": "content-and-config",
    "webarena.437": "content-and-config",
    "webarena.438": "content-and-config",
    "webarena.439": "content-and-config",
    "webarena.440": "content-and-config",
    "webarena.441": "content-and-config",
    "webarena.442": "content-and-config",
    "webarena.443": "content-and-config",
    "webarena.444": "content-and-config",
    "webarena.445": "content-and-config",
    "webarena.446": "content-and-config",
    "webarena.447": "content-and-config",
    "webarena.448": "content-and-config",
    "webarena.449": "content-and-config",
    "webarena.450": "content-and-config",
    "webarena.451": "content-and-config",
    "webarena.452": "content-and-config",
    "webarena.453": "content-and-config",
    "webarena.454": "content-and-config",
    "webarena.455": "content-and-config",
    "webarena.456": "content-and-config",
    "webarena.457": "content-and-config",
    "webarena.458": "content-and-config",
    "webarena.459": "content-and-config",
    "webarena.460": "content-and-config",
    "webarena.461": "content-and-config",
    "webarena.462": "content-and-config",
    "webarena.463": "content-and-config",
    "webarena.464": "content-and-config",
    "webarena.465": "content-and-config",
    "webarena.466": "content-and-config",
    "webarena.467": "content-and-config",
    "webarena.468": "content-and-config",
    "webarena.469": "content-and-config",
    "webarena.470": "content-and-config",
    "webarena.471": "content-and-config",
    "webarena.472": "content-and-config",
    "webarena.473": "content-and-config",
    "webarena.474": "content-and-config",
    "webarena.475": "content-and-config",
    "webarena.476": "content-and-config",
    "webarena.477": "content-and-config",
    "webarena.478": "content-and-config",
    "webarena.479": "content-and-config",
    "webarena.480": "content-and-config",
    "webarena.481": "content-and-config",
    "webarena.482": "content-and-config",
    "webarena.483": "content-and-config",
    "webarena.484": "content-and-config",
    "webarena.485": "content-and-config",
    "webarena.486": "content-and-config",
    "webarena.487": "content-and-config",
    "webarena.488": "content-and-config",
    "webarena.489": "content-and-config",
    "webarena.490": "content-and-config",
    "webarena.491": "information-seeking",
    "webarena.492": "content-and-config",
    "webarena.493": "content-and-config",
    "webarena.494": "content-and-config",
    "webarena.495": "content-and-config",
    "webarena.496": "content-and-config",
    "webarena.497": "content-and-config",
    "webarena.498": "content-and-config",
    "webarena.499": "content-and-config",
    "webarena.500": "content-and-config",
    "webarena.501": "content-and-config",
    "webarena.502": "content-and-config",
    "webarena.503": "content-and-config",
    "webarena.504": "content-and-config",
    "webarena.505": "content-and-config",
    "webarena.506": "content-and-config",
    "webarena.507": "content-and-config",
    "webarena.508": "content-and-config",
    "webarena.509": "content-and-config",
    "webarena.510": "content-and-config",
    "webarena.511": "content-and-config",
    "webarena.512": "content-and-config",
    "webarena.513": "content-and-config",
    "webarena.514": "content-and-config",
    "webarena.515": "content-and-config",
    "webarena.516": "content-and-config",
    "webarena.517": "content-and-config",
    "webarena.518": "content-and-config",
    "webarena.519": "content-and-config",
    "webarena.520": "content-and-config",
    "webarena.521": "content-and-config",
    "webarena.522": "content-and-config",
    "webarena.523": "content-and-config",
    "webarena.524": "content-and-config",
    "webarena.525": "content-and-config",
    "webarena.526": "content-and-config",
    "webarena.527": "content-and-config",
    "webarena.528": "content-and-config",
    "webarena.529": "content-and-config",
    "webarena.530": "content-and-config",
    "webarena.531": "content-and-config",
    "webarena.532": "content-and-config",
    "webarena.533": "content-and-config",
    "webarena.534": "content-and-config",
    "webarena.535": "content-and-config",
    "webarena.536": "content-and-config",
    "webarena.537": "content-and-config",
    "webarena.538": "content-and-config",
    "webarena.539": "content-and-config",
    "webarena.540": "content-and-config",
    "webarena.541": "content-and-config",
    "webarena.542": "content-and-config",
    "webarena.543": "content-and-config",
    "webarena.544": "content-and-config",
    "webarena.545": "content-and-config",
    "webarena.546": "content-and-config",
    "webarena.547": "content-and-config",
    "webarena.548": "content-and-config",
    "webarena.549": "content-and-config",
    "webarena.550": "content-and-config",
    "webarena.551": "content-and-config",
    "webarena.552": "content-and-config",
    "webarena.553": "content-and-config",
    "webarena.554": "content-and-config",
    "webarena.555": "content-and-config",
    "webarena.556": "content-and-config",
    "webarena.557": "content-and-config",
    "webarena.558": "content-and-config",
    "webarena.559": "content-and-config",
    "webarena.560": "content-and-config",
    "webarena.561": "content-and-config",
    "webarena.562": "content-and-config",
    "webarena.563": "content-and-config",
    "webarena.564": "content-and-config",
    "webarena.565": "content-and-config",
    "webarena.566": "content-and-config",
    "webarena.567": "content-and-config",
    "webarena.568": "content-and-config",
    "webarena.569": "content-and-config",
    "webarena.570": "content-and-config",
    "webarena.571": "content-and-config",
    "webarena.572": "content-and-config",
    "webarena.573": "content-and-config",
    "webarena.574": "content-and-config",
    "webarena.575": "content-and-config",
    "webarena.576": "content-and-config",
    "webarena.577": "content-and-config",
    "webarena.578": "content-and-config",
    "webarena.579": "content-and-config",
    "webarena.580": "content-and-config",
    "webarena.581": "content-and-config",
    "webarena.582": "content-and-config",
    "webarena.583": "content-and-config",
    "webarena.584": "content-and-config",
    "webarena.585": "content-and-config",
    "webarena.586": "content-and-config",
    "webarena.587": "content-and-config",
    "webarena.588": "content-and-config",
    "webarena.589": "content-and-config",
    "webarena.590": "content-and-config",
    "webarena.591": "content-and-config",
    "webarena.592": "content-and-config",
    "webarena.593": "content-and-config",
    "webarena.594": "content-and-config",
    "webarena.595": "content-and-config",
    "webarena.596": "content-and-config",
    "webarena.597": "content-and-config",
    "webarena.598": "content-and-config",
    "webarena.599": "content-and-config",
    "webarena.600": "content-and-config",
    "webarena.601": "content-and-config",
    "webarena.602": "content-and-config",
    "webarena.603": "content-and-config",
    "webarena.604": "content-and-config",
    "webarena.605": "content-and-config",
    "webarena.606": "content-and-config",
    "webarena.607": "content-and-config",
    "webarena.608": "content-and-config",
    "webarena.609": "content-and-config",
    "webarena.610": "content-and-config",
    "webarena.611": "content-and-config",
    "webarena.612": "content-and-config",
    "webarena.613": "content-and-config",
    "webarena.614": "content-and-config",
    "webarena.615": "content-and-config",
    "webarena.616": "content-and-config",
    "webarena.617": "content-and-config",
    "webarena.618": "content-and-config",
    "webarena.619": "content-and-config",
    "webarena.620": "content-and-config",
    "webarena.621": "content-and-config",
    "webarena.622": "content-and-config",
    "webarena.623": "content-and-config",
    "webarena.624": "content-and-config",
    "webarena.625": "content-and-config",
    "webarena.626": "content-and-config",
    "webarena.627": "content-and-config",
    "webarena.628": "content-and-config",
    "webarena.629": "content-and-config",
    "webarena.630": "content-and-config",
    "webarena.631": "content-and-config",
    "webarena.632": "content-and-config",
    "webarena.633": "content-and-config",
    "webarena.634": "content-and-config",
    "webarena.635": "content-and-config",
    "webarena.636": "content-and-config",
    "webarena.637": "content-and-config",
    "webarena.638": "content-and-config",
    "webarena.639": "content-and-config",
    "webarena.640": "content-and-config",
    "webarena.641": "content-and-config",
    "webarena.642": "content-and-config",
    "webarena.643": "content-and-config",
    "webarena.644": "content-and-config",
    "webarena.645": "content-and-config",
    "webarena.646": "content-and-config",
    "webarena.647": "content-and-config",
    "webarena.648": "content-and-config",
    "webarena.649": "content-and-config",
    "webarena.650": "content-and-config",
    "webarena.651": "content-and-config",
    "webarena.652": "content-and-config",
    "webarena.653": "content-and-config",
    "webarena.654": "content-and-config",
    "webarena.655": "content-and-config",
    "webarena.656": "content-and-config",
    "webarena.657": "content-and-config",
    "webarena.658": "content-and-config",
    "webarena.659": "content-and-config",
    "webarena.660": "content-and-config",
    "webarena.661": "content-and-config",
    "webarena.662": "content-and-config",
    "webarena.663": "content-and-config",
    "webarena.664": "content-and-config",
    "webarena.665": "content-and-config",
    "webarena.666": "content-and-config",
    "webarena.667": "content-and-config",
    "webarena.668": "content-and-config",
    "webarena.669": "content-and-config",
    "webarena.670": "content-and-config",
    "webarena.671": "content-and-config",
    "webarena.672": "content-and-config",
    "webarena.673": "content-and-config",
    "webarena.674": "content-and-config",
    "webarena.675": "content-and-config",
    "webarena.676": "content-and-config",
    "webarena.677": "content-and-config",
    "webarena.678": "content-and-config",
    "webarena.679": "content-and-config",
    "webarena.680": "content-and-config",
    "webarena.681": "content-and-config",
    "webarena.682": "content-and-config",
    "webarena.683": "content-and-config",
    "webarena.684": "content-and-config",
    "webarena.685": "content-and-config",
    "webarena.686": "content-and-config",
    "webarena.687": "content-and-config",
    "webarena.688": "content-and-config",
    "webarena.689": "content-and-config",
    "webarena.690": "content-and-config",
    "webarena.691": "content-and-config",
    "webarena.692": "content-and-config",
    "webarena.693": "content-and-config",
    "webarena.694": "content-and-config",
    "webarena.695": "content-and-config",
    "webarena.696": "content-and-config",
    "webarena.697": "content-and-config",
    "webarena.698": "content-and-config",
    "webarena.699": "content-and-config",
    "webarena.700": "content-and-config",
    "webarena.701": "content-and-config",
    "webarena.702": "content-and-config",
    "webarena.703": "content-and-config",
    "webarena.704": "content-and-config",
    "webarena.705": "content-and-config",
    "webarena.706": "content-and-config",
    "webarena.707": "content-and-config",
    "webarena.708": "content-and-config",
    "webarena.709": "content-and-config",
    "webarena.710": "content-and-config",
    "webarena.711": "content-and-config",
    "webarena.712": "content-and-config",
    "webarena.713": "content-and-config",
    "webarena.714": "content-and-config",
    "webarena.715": "content-and-config",
    "webarena.716": "content-and-config",
    "webarena.717": "content-and-config",
    "webarena.718": "content-and-config",
    "webarena.719": "content-and-config",
    "webarena.720": "content-and-config",
    "webarena.721": "content-and-config",
    "webarena.722": "content-and-config",
    "webarena.723": "information-seeking",
    "webarena.724": "content-and-config",
    "webarena.725": "content-and-config",
    "webarena.726": "information-seeking",
    "webarena.727": "content-and-config",
    "webarena.728": "content-and-config",
    "webarena.729": "content-and-config",
    "webarena.730": "content-and-config",
    "webarena.731": "content-and-config",
    "webarena.732": "content-and-config",
    "webarena.733": "content-and-config",
    "webarena.734": "content-and-config",
    "webarena.735": "content-and-config",
    "webarena.736": "content-and-config",
    "webarena.737": "content-and-config",
    "webarena.738": "content-and-config",
    "webarena.739": "content-and-config",
    "webarena.740": "content-and-config",
    "webarena.741": "content-and-config",
    "webarena.742": "content-and-config",
    "webarena.743": "content-and-config",
    "webarena.744": "content-and-config",
    "webarena.745": "content-and-config",
    "webarena.746": "content-and-config",
    "webarena.747": "content-and-config",
    "webarena.748": "content-and-config",
    "webarena.749": "content-and-config",
    "webarena.750": "content-and-config",
    "webarena.751": "content-and-config",
    "webarena.752": "content-and-config",
    "webarena.753": "content-and-config",
    "webarena.754": "content-and-config",
    "webarena.755": "content-and-config",
    "webarena.756": "content-and-config",
    "webarena.757": "content-and-config",
    "webarena.758": "content-and-config",
    "webarena.759": "content-and-config",
    "webarena.760": "content-and-config",
    "webarena.761": "content-and-config",
    "webarena.762": "content-and-config",
    "webarena.763": "content-and-config",
    "webarena.764": "content-and-config",
    "webarena.765": "content-and-config",
    "webarena.766": "content-and-config",
    "webarena.767": "content-and-config",
    "webarena.768": "content-and-config",
    "webarena.769": "content-and-config",
    "webarena.770": "content-and-config",
    "webarena.771": "content-and-config",
    "webarena.772": "content-and-config",
    "webarena.773": "content-and-config",
    "webarena.774": "content-and-config",
    "webarena.775": "content-and-config",
    "webarena.776": "content-and-config",
    "webarena.777": "content-and-config",
    "webarena.778": "content-and-config",
    "webarena.779": "content-and-config",
    "webarena.780": "content-and-config",
    "webarena.781": "content-and-config",
    "webarena.782": "content-and-config",
    "webarena.783": "information-seeking",
    "webarena.784": "information-seeking",
    "webarena.785": "information-seeking",
    "webarena.786": "information-seeking",
    "webarena.787": "information-seeking",
    "webarena.788": "information-seeking",
    "webarena.789": "information-seeking",
    "webarena.790": "information-seeking",
    "webarena.791": "information-seeking",
    "webarena.792": "information-seeking",
    "webarena.793": "information-seeking",
    "webarena.794": "information-seeking",
    "webarena.795": "information-seeking",
    "webarena.796": "information-seeking",
    "webarena.797": "information-seeking",
    "webarena.798": "information-seeking",
    "webarena.799": "content-and-config",
    "webarena.800": "content-and-config",
    "webarena.801": "content-and-config",
    "webarena.802": "content-and-config",
    "webarena.803": "content-and-config",
    "webarena.804": "content-and-config",
    "webarena.805": "content-and-config",
    "webarena.806": "content-and-config",
    "webarena.807": "content-and-config",
    "webarena.808": "content-and-config",
    "webarena.809": "content-and-config",
    "webarena.810": "content-and-config",
    "webarena.811": "content-and-config",
}


def make_joint_ablation_study(result_dict):
    """Generate an ablation report for all models."""
    col_dict = {}
    for model_name, result_df in result_dict.items():
        report = inspect_results.ablation_report(result_df)
        short_model_name = NAME_MAP.get(model_name, model_name)
        col_dict[short_model_name] = 100 * report["avg_reward"]
        col_dict[f"±{short_model_name}"] = 100 * report["uncertainty_reward"]

    return pd.DataFrame(col_dict)


def make_joint_report(result_dict, agent_index_dict=None, use_category=True):
    """Select a specific agent and generate a report for all models.

    Args:
        result_dict (dict): a dictionary of dataframes for each benchmark
        agent_index_dict (dict): a dictionary of agent index. If a single index
            is used, it will be used for all benchmarks
        use_category (bool): if True, use the task category as index. Otherwise,
            will return the report for all tasks.

    Returns:
        pd.DataFrame: a dataframe with the average reward and uncertainty for
            each model.
    """
    col_dict = {}
    for model_name, result_df in result_dict.items():
        if agent_index_dict is not None:
            if isinstance(agent_index_dict, dict):
                agent_index = agent_index_dict[model_name]
            else:
                agent_index = agent_index_dict
            agent_result_df = select_single_agent(result_df, agent_index)
        else:
            agent_result_df = result_df
        if use_category:
            agent_result_df = inspect_results.set_task_category_as_index(agent_result_df)
        report = inspect_results.global_report(agent_result_df, rename_index=None)
        short_model_name = NAME_MAP.get(model_name, model_name)
        col_dict[short_model_name] = 100 * report["avg_reward"]
        col_dict[f"±{short_model_name}"] = 100 * report["uncertainty_reward"]

    return pd.DataFrame(col_dict)


def make_joint_ablation(result_dict):
    """Select a specific agent and generate a report for all models.

    Args:
        result_dict (dict): a dictionary of dataframes for each benchmark
        agent_index_dict (dict): a dictionary of agent index. If a single index
            is used, it will be used for all benchmarks
        use_category (bool): if True, use the task category as index. Otherwise,
            will return the report for all tasks.

    Returns:
        pd.DataFrame: a dataframe with the average reward and uncertainty for
            each model.
    """
    col_dict = {}
    for name, result_df in result_dict.items():

        report = inspect_results.ablation_report(result_df)
        short_name = NAME_MAP.get(name, name)
        col_dict[short_name] = 100 * report["avg_reward"]
        col_dict[f"±{short_name}"] = 100 * report["uncertainty_reward"]

    return pd.DataFrame(col_dict)


def add_web_gum_subset(result_df):
    """Add the webgum subset to the result_df"""
    webgum_df = result_df[
        result_df.index.get_level_values("env_args.task_name").isin(webgum_tasks)
    ].copy()

    webgum_df["task_category"] = "webgum"
    result_df["task_category"] = "all"
    return pd.concat([result_df, webgum_df])


def paper_stats(sub_df, quantile=0.95):
    """Extract stats to generate plot of the paper"""
    record = {
        "max DOM tokens": sub_df["stats.max_token_dom_txt"].quantile(quantile),
        "max Pruned DOM tokens": sub_df["stats.max_token_pruned_html"].quantile(quantile),
        "max AXTree tokens": sub_df["stats.max_token_axtree_txt"].quantile(quantile),
        "episode DOM tokens": sub_df["stats.cum_token_dom_txt"].mean(),
        "episode Pruned DOM tokens": sub_df["stats.cum_token_pruned_html"].mean(),
        "episode AXTree tokens": sub_df["stats.cum_token_axtree_txt"].mean(),
    }

    return pd.Series(record)


def step_action_count(action_str: str):
    """Count the number of actions in a step from an action string as parsed by
    highlevel_action_parser in browsergym."""
    function_calls = sum(highlevel_action_parser.search_string(action_str))
    if isinstance(function_calls, int):
        return function_calls
    else:
        return len(function_calls)


def episode_action_count(exp_result: ExpResult):
    """Count the number of actions in an episode, including multiple actions in
    one step."""
    episode = exp_result.steps_info
    return sum([step_action_count(step_info.action) for step_info in episode])


cache_dir = str((Path.home() / ".agentlab_cache").mkdir(exist_ok=True))
memory = Memory(cache_dir, verbose=0)


def filter_multi_action_and_sucess(exp_result: ExpResult):
    """Only keep experiments that have multi_actions and are successful."""
    info = exp_result.get_exp_record()
    try:
        success = info["cum_reward"] > 0
        return success and info["agent_args.flags.multi_actions"]
    except KeyError:
        return False


@memory.cache
def get_all_action_count(exp_dir: str | Path, filter_func=filter_multi_action_and_sucess):
    """Extract the number of actions for each episode for all experiments in a
    directory.

    Args:
        exp_dir (str | Path): Recursively search experiments from this directory.
        filter_func (function): A callable returning False if the experiment
            should be skipped.

    Returns:
        pd.DataFrame: as defined by ExpResults.get_summary, but with an added
        column n_actions.
    """
    all_results = list(yield_all_exp_results(exp_dir, use_cache=False))

    info = []
    for exp_result in tqdm(all_results):
        if not filter_func(exp_result):
            continue
        n_actions = episode_action_count(exp_result)
        summary = exp_result.get_exp_record()

        summary["n_actions"] = n_actions
        info.append(summary)
    return pd.DataFrame(info)


##########
# These are prompt to help generating the latex tables for the paper. Just
# update the results in the prompt and ask GPT to generate the latex table.

_prompt_for_main_table = r"""

Here is my current table,

---------------

% Define a command for the smaller, gray-scale text
\newcommand{\gpm}[1]{\textcolor{gray}{\tiny$\pm$#1}}

% New column types for left and right alignment
\newcolumntype{L}{>{\raggedright\arraybackslash}X}
\newcolumntype{R}{>{\raggedleft\arraybackslash}X}

\newcommand{\warning}{\textcolor{orange}{\faExclamationTriangle{}}}
\newcommand{\warningNote}{\warning\; \textcolor{blue}{WorkArena has changed on GitHub and those results are obtained using an older version. Reproducible results will be published in an updated version of the paper.}\;\warning}

\begin{table}[t] % Use table for single column
\caption{Success rate\gpm{Standard error} (SR \gpm{SE}) of all agents on MiniWoB, WorkArena, and WebArena. Bolded numbers represent the average success rate over the entire corresponding benchmark.}
\noindent\resizebox{\columnwidth}{!}{ % Adjust to column width
\begin{tabular}{l r@{\hspace{2pt}}l r@{\hspace{2pt}}l r@{\hspace{2pt}}l r@{\hspace{2pt}}l r@{\hspace{2pt}}l}
\toprule
\textbf{Task Category} & \multicolumn{2}{c}{\textbf{GPT-4o}} & \multicolumn{2}{c}{\textbf{GPT-4o-v}} & \multicolumn{2}{c}{\textbf{GPT-3.5}} & \multicolumn{2}{c}{\textbf{Llama3}} \\
 & \textbf{SR \%} & \gpm{SE} & \textbf{SR \%} & \gpm{SE} & \textbf{SR \%} & \gpm{SE} & \textbf{SR \%} & \gpm{SE} \\
\midrule
\textbf{WorkArena} & \textbf{42.7} & \gpm{1.7} & \textbf{41.8} & \gpm{1.8} & \textbf{18.6} & \gpm{2.2} & \textbf{0} & \gpm{0} \\
\quad Dashboard & 62.5 & \gpm{7.2} & 72.5 & \gpm{6.4} & 20.0 & \gpm{5.8} & 0 & \gpm{0} \\
\quad Form & 40.0 & \gpm{6.2} & 34.0 & \gpm{4.9} & 2.0 & \gpm{2.3} & 0 & \gpm{0} \\
\quad Knowledge & 80.0 & \gpm{12.4} & 70.0 & \gpm{14.1} & 0.0 & \gpm{4.3} & 0 & \gpm{0} \\
\quad List-filter & 0.0 & \gpm{1.9} & 0.0 & \gpm{1.5} & 0.0 & \gpm{1.6} & 0 & \gpm{0} \\
\quad List-sort & 10.0 & \gpm{3.4} & 13.3 & \gpm{3.9} & 8.3 & \gpm{3.6} & 0 & \gpm{0} \\
\quad Menu & 60.0 & \gpm{7.9} & 90.0 & \gpm{6.0} & 5.0 & \gpm{4.4} & 0 & \gpm{0} \\
\quad Service catalog & 77.8 & \gpm{3.0} & 65.6 & \gpm{4.2} & 5.6 & \gpm{2.5} & 0 & \gpm{0} \\
\midrule
\textbf{MiniWoB} {\tiny(125 tasks)} & \textbf{66.1} & \gpm{1.0} & \textbf{67.7} & \gpm{1.0} & \textbf{38.9} & \gpm{1.0} & \textbf{25.5} & \gpm{1.3} \\
\quad WebGum Subset {\tiny(56 tasks)} & 82.9 & \gpm{1.3} & 83.2 & \gpm{1.3} & 53.6 & \gpm{1.6} & 32.4 & \gpm{2.1} \\
\midrule
\textbf{WebArena} {\tiny(812 tasks)} & \textbf{24.0} & \gpm{0.7} & \textbf{23.5} & \gpm{0.6} & \textbf{13.1} & \gpm{0.6} & \textbf{0} & \gpm{0} \\
\quad Content-and-config {\tiny(411 tasks)} & 26.8 & \gpm{0.9} & 25.8 & \gpm{0.8} & 8.8 & \gpm{0.8} & 0 & \gpm{0} \\
\quad Information-seeking {\tiny(325 tasks)} & 22.5 & \gpm{1.1} & 22.5 & \gpm{1.0} & 4.3 & \gpm{0.9} & 0 & \gpm{0} \\
\quad Navigation {\tiny(76 tasks)} & 15.8 & \gpm{1.7} & 15.8 & \gpm{2.1} & 5.3 & \gpm{1.9} & 0 & \gpm{0} \\
\bottomrule
\end{tabular}
}
\label{tab:acc-summary}
\end{table}
-----------

I need to update with new results:
Only the following values are the good ones. The previous values are just
placeholders. If there are missing values put N/A. [ALL TASKS] corresponds to
the main results for the benchmark. 

MiniWob:
	Llama3	±Llama3	gpt-3.5	±gpt-3.5	gpt-4o	±gpt-4o	gpt-4o-v	±gpt-4o-v
task_category								
all	62.6	0.6	38.9	1.1	66.1	1.0	67.7	1.0
webgum	80.5	1.0	53.6	1.4	82.9	1.5	83.2	1.5

WorkArena Results:
gpt-3.5	±gpt-3.5	gpt-4o	±gpt-4o	gpt-4o-v	±gpt-4o-v	Llama3	±Llama3
task_category								
dashboard	20.0	4.8	62.5	6.8	72.5	6.0	37.5	6.0
form	2.0	2.5	40.0	5.9	34.0	4.8	32.0	4.6
knowledge	0.0	4.3	80.0	12.2	70.0	13.9	30.0	12.3
list-filter	0.0	1.6	0.0	1.6	0.0	1.7	0.0	1.8
list-sort	8.3	3.7	10.0	3.8	13.3	4.0	1.7	2.5
menu	5.0	4.7	60.0	8.0	90.0	6.0	0.0	2.9
service catalog	5.6	2.3	77.8	3.2	65.6	3.6	26.7	3.4
[ALL TASKS]	6.1	1.3	42.7	1.5	41.8	1.7	17.9	1.5

WebArena Results:
gpt-3.5	±gpt-3.5	gpt-4o	±gpt-4o	gpt-4o-v	±gpt-4o-v	Llama3	±Llama3
task_category								
content-and-config	8.8	0.8	26.8	0.9	25.8	1.0	12.7	0.9
information-seeking	4.3	0.9	22.5	0.9	22.5	1.0	9.8	1.1
navigation	5.3	1.9	15.8	1.8	15.8	2.2	6.6	1.9
[ALL TASKS]	6.7	0.6	24.0	0.6	23.5	0.7	11.0	0.6
--------

"""


_prompt_for_ablation_table = r"""


Getting inspiration from this table,


% Define a command for the smaller, gray-scale text
\newcommand{\gpm}[1]{\textcolor{gray}{\small$\pm#1$}}

\begin{table}[ht]
\caption{Ablation study on \textbf{MiniWoB++} and \textbf{workarena}. Success rate\gpm{Standard error} (SR \gpm{SE}) of all configurations. Each row modifies the initial configuration.}
\noindent\resizebox{\columnwidth}{!}{
\begin{tabular}{l r@{\hspace{2pt}}l r@{\hspace{2pt}}l}
\toprule
\multicolumn{1}{l}{\textbf{Configuration}} & \multicolumn{2}{c}{\textbf{MiniWoB++}} & \multicolumn{2}{c}{\textbf{workarena}} \\
 & \textbf{SR \%} & \gpm{SE} & \textbf{SR \%} & \gpm{SE} \\
\midrule
Initial Configuration & 68.2 & \gpm{1.0} & 45.5 & \gpm{2.2} \\
+multi\_actions & 68.5 & \gpm{1.0} & 40.6 & \gpm{2.0} \\
+long\_description & 67.8 & \gpm{1.0} & 45.5 & \gpm{2.3} \\
+individual\_examples & 67.0 & \gpm{1.0} & 43.0 & \gpm{2.0} \\
action\_set← bid+coord with box & 72.6 & \gpm{1.0} & 41.2 & \gpm{1.8} \\
+use\_think\_history & 66.7 & \gpm{0.9} & 42.4 & \gpm{2.3} \\
+use\_past\_error\_logs & 67.2 & \gpm{0.9} & 43.6 & \gpm{2.1} \\
-extract\_visible\_tag & 68.8 & \gpm{1.0} & 43.0 & \gpm{2.2} \\
\bottomrule
\end{tabular}
}
\label{tab:ablation-mw-workarena}
\end{table}

-------------

Create a latex table with those new results by getting the template of the
previous table.
Remove redundant "obs." and "action." prefixes from the configuration names.
use N/A instead of NaN

MiniWoB++	±MiniWoB++	workarena	±workarena
change				
+action.long_description	55.5	1.0	17.6	2.0
+action.multi_actions	63.0	1.0	17.6	2.1
+obs.use_error_logs	60.2	1.1	19.4	1.8
+use_plan	59.4	0.9	NaN	NaN
-obs.extract_visible_tag	61.3	0.9	15.8	2.0
-obs.use_action_history	49.9	1.1	8.5	1.7
-obs.use_think_history	52.2	1.1	18.8	2.1
-use_abstract_example	59.4	1.1	NaN	NaN
-use_concrete_example	59.4	1.1	NaN	NaN
-use_thinking	48.6	1.1	8.5	1.7
Initial Configuration	59.8	1.0	20.0	2.3
"""


main_table_neurips = r"""


get inspiration from the format of the original table and write a new latex
tables contaiing the new results. Make sure to copy only the value of the new
results and put values in the appropriate column order for each sub table. 
rows starting with [name] are main categories and should be bold. Other rows are
sub categories and should start with \quad

Here is my old table,

---------------

% Define a command for the smaller, gray-scale text
\newcommand{\gpm}[1]{\textcolor{gray}{\tiny$\pm$#1}}

% New column types for left and right alignment
\newcolumntype{L}{>{\raggedright\arraybackslash}X}
\newcolumntype{R}{>{\raggedleft\arraybackslash}X}

\begin{table}[t] % Use table for single column
\caption{Success rate\gpm{Standard error} (SR \gpm{SE}) of all agents on MiniWoB, WorkArena, and WebArena. Bolded numbers represent the average success rate over the entire corresponding benchmark.}
\noindent\resizebox{\columnwidth}{!}{ % Adjust to column width
\begin{tabular}{l r@{\hspace{2pt}}l r@{\hspace{2pt}}l r@{\hspace{2pt}}l r@{\hspace{2pt}}l r@{\hspace{2pt}}l}
\toprule
\textbf{Task Category} & \multicolumn{2}{c}{\textbf{GPT-4o}} & \multicolumn{2}{c}{\textbf{GPT-4o-v}} & \multicolumn{2}{c}{\textbf{GPT-3.5}} & \multicolumn{2}{c}{\textbf{Llama3}} \\
 & \textbf{SR \%} & \gpm{SE} & \textbf{SR \%} & \gpm{SE} & \textbf{SR \%} & \gpm{SE} & \textbf{SR \%} & \gpm{SE} \\
\midrule
\textbf{WorkArena L3} & \textbf{0.0} & \gpm{0.0} & \textbf{0.0} & \gpm{0.0} & \textbf{0.0} & \gpm{0.0} & \textbf{0.0} & \gpm{0.0} \\
\quad Infeasible tasks & 0.0 & \gpm{0.0} & 0.0 & \gpm{0.0} & 0.0 & \gpm{0.0} & 0.0 & \gpm{0.0} \\
\quad Logical and mathematical reasoning & 0.0 & \gpm{0.0} & 0.0 & \gpm{0.0} & 0.0 & \gpm{0.0} & 0.0 & \gpm{0.0} \\
\quad Planning, decision making, and problem solving & 0.0 & \gpm{0.0} & 0.0 & \gpm{0.0} & 0.0 & \gpm{0.0} & 0.0 & \gpm{0.0} \\
\quad Retrieval & 0.0 & \gpm{0.0} & 0.0 & \gpm{0.0} & 0.0 & \gpm{0.0} & 0.0 & \gpm{0.0} \\
\quad Sophisticated memory & 0.0 & \gpm{0.0} & 0.0 & \gpm{0.0} & 0.0 & \gpm{0.0} & 0.0 & \gpm{0.0} \\
\midrule
\textbf{WorkArena L2} & \textbf{3.0} & \gpm{1.1} & \textbf{3.8} & \gpm{1.3} & \textbf{0.0} & \gpm{0.0} & \textbf{0.0} & \gpm{0.0} \\
\quad Infeasible tasks & 0.0 & \gpm{0.0} & 0.0 & \gpm{0.0} & 0.0 & \gpm{0.0} & 0.0 & \gpm{0.0} \\
\quad Logical and mathematical reasoning & 0.0 & \gpm{0.0} & 0.0 & \gpm{0.0} & 0.0 & \gpm{0.0} & 0.0 & \gpm{0.0} \\
\quad Planning, decision making, and problem solving & 0.0 & \gpm{0.0} & 0.0 & \gpm{0.0} & 0.0 & \gpm{0.0} & 0.0 & \gpm{0.0} \\
\quad Retrieval & 0.0 & \gpm{0.0} & 3.6 & \gpm{2.5} & 0.0 & \gpm{0.0} & 0.0 & \gpm{0.0} \\
\quad Sophisticated memory & 14.6 & \gpm{5.1} & 14.6 & \gpm{5.1} & 0.0 & \gpm{0.0} & 0.0 & \gpm{0.0} \\
\midrule
\textbf{WorkArena L1} & \textbf{42.7} & \gpm{2.7} & \textbf{41.8} & \gpm{2.7} & \textbf{17.9} & \gpm{2.1} & \textbf{6.1} & \gpm{1.3} \\
\midrule
\textbf{MiniWoB} & \textbf{71.3} & \gpm{1.5} & \textbf{72.5} & \gpm{1.5} & \textbf{68.2} & \gpm{1.2} & \textbf{43.4} & \gpm{1.6} \\
\midrule
\textbf{WebArena} & \textbf{24.0} & \gpm{1.5} & \textbf{23.5} & \gpm{1.5} & \textbf{11.0} & \gpm{1.1} & \textbf{6.7} & \gpm{0.9} \\
\bottomrule
\end{tabular}
}
\label{tab:acc-summary}
\end{table}

-----------


Llama3	±Llama3	gpt-3.5	±gpt-3.5	gpt-4o	±gpt-4o	gpt-4o-v	±gpt-4o-v	mixtral	±mixtral
[WorkArena L3]	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
infeasible_tasks	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
logical_and_mathematical_reasoning	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
planning_decision_making_and_problem_solving	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
retrieval	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
sophisticated_memory	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
[WorkArena L2]	0.0	0.0	0.0	0.0	3.0	1.1	3.8	1.3	0.0	0.0
infeasible_tasks	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
logical_and_mathematical_reasoning	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
planning_decision_making_and_problem_solving	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
retrieval	0.0	0.0	0.0	0.0	0.0	0.0	3.6	2.5	0.0	0.0
sophisticated_memory	0.0	0.0	0.0	0.0	14.6	5.1	14.6	5.1	0.0	0.0
[WorkArena L1]	17.9	2.1	6.1	1.3	42.7	2.7	41.8	2.7	12.4	1.8
[MiniWob]	68.2	1.2	43.4	1.6	71.3	1.5	72.5	1.5	62.4	1.6
[WebArena]	11.0	1.1	6.7	0.9	24.0	1.5	23.5	1.5	NaN	NaN
-------------


"""


human_eval = r"""

convert the data below into a latex table. the "se" columns are for standard
error. use the \gpm command to display the standard deviation in gray color
in the same column as the value. Level 2 and Level 3 represents grouped columns
headers. All tasks should be in bold and other categories should be preceded by \quad

here is another table for the style:

% Define a command for the smaller, gray-scale text
\newcommand{\gpm}[1]{\textcolor{gray}{\tiny$\pm$#1}}

% New column types for left and right alignment
\newcolumntype{L}{>{\raggedright\arraybackslash}X}
\newcolumntype{R}{>{\raggedleft\arraybackslash}X}

\newcommand{\warning}{\textcolor{orange}{\faExclamationTriangle{}}}
\newcommand{\warningNote}{\warning\; \textcolor{blue}{WorkArena has changed on GitHub and those results are obtained using an older version. Reproducible results will be published in an updated version of the paper.}\;\warning}

\begin{table}[t] % Use table for single column
\caption{Success rate\gpm{Standard error} (SR \gpm{SE}) of all agents on MiniWoB, WorkArena, and WebArena. Bolded numbers represent the average success rate over the entire corresponding benchmark.}
\noindent\resizebox{\columnwidth}{!}{ % Adjust to column width
\begin{tabular}{l r@{\hspace{2pt}}l r@{\hspace{2pt}}l r@{\hspace{2pt}}l r@{\hspace{2pt}}l r@{\hspace{2pt}}l}
\toprule
\textbf{Task Category} & \multicolumn{2}{c}{\textbf{GPT-4o}} & \multicolumn{2}{c}{\textbf{GPT-4o-v}} & \multicolumn{2}{c}{\textbf{GPT-3.5}} & \multicolumn{2}{c}{\textbf{Llama3}} \\
 & \textbf{SR \%} & \gpm{SE} & \textbf{SR \%} & \gpm{SE} & \textbf{SR \%} & \gpm{SE} & \textbf{SR \%} & \gpm{SE} \\
\midrule
\textbf{WorkArena} & \textbf{42.7} & \gpm{1.7} & \textbf{41.8} & \gpm{1.8} & \textbf{18.6} & \gpm{2.2} & \textbf{0} & \gpm{0} \\
\quad Dashboard & 62.5 & \gpm{7.2} & 72.5 & \gpm{6.4} & 20.0 & \gpm{5.8} & 0 & \gpm{0} \\
\quad Form & 40.0 & \gpm{6.2} & 34.0 & \gpm{4.9} & 2.0 & \gpm{2.3} & 0 & \gpm{0} \\
\quad Knowledge & 80.0 & \gpm{12.4} & 70.0 & \gpm{14.1} & 0.0 & \gpm{4.3} & 0 & \gpm{0} \\
\quad List-filter & 0.0 & \gpm{1.9} & 0.0 & \gpm{1.5} & 0.0 & \gpm{1.6} & 0 & \gpm{0} \\
\quad List-sort & 10.0 & \gpm{3.4} & 13.3 & \gpm{3.9} & 8.3 & \gpm{3.6} & 0 & \gpm{0} \\
\quad Menu & 60.0 & \gpm{7.9} & 90.0 & \gpm{6.0} & 5.0 & \gpm{4.4} & 0 & \gpm{0} \\
\quad Service catalog & 77.8 & \gpm{3.0} & 65.6 & \gpm{4.2} & 5.6 & \gpm{2.5} & 0 & \gpm{0} \\
\midrule
\textbf{MiniWoB} {\tiny(125 tasks)} & \textbf{66.1} & \gpm{1.0} & \textbf{67.7} & \gpm{1.0} & \textbf{38.9} & \gpm{1.0} & \textbf{25.5} & \gpm{1.3} \\
\quad WebGum Subset {\tiny(56 tasks)} & 82.9 & \gpm{1.3} & 83.2 & \gpm{1.3} & 53.6 & \gpm{1.6} & 32.4 & \gpm{2.1} \\
\midrule
\textbf{WebArena} {\tiny(812 tasks)} & \textbf{24.0} & \gpm{0.7} & \textbf{23.5} & \gpm{0.6} & \textbf{13.1} & \gpm{0.6} & \textbf{0} & \gpm{0} \\
\quad Content-and-config {\tiny(411 tasks)} & 26.8 & \gpm{0.9} & 25.8 & \gpm{0.8} & 8.8 & \gpm{0.8} & 0 & \gpm{0} \\
\quad Information-seeking {\tiny(325 tasks)} & 22.5 & \gpm{1.1} & 22.5 & \gpm{1.0} & 4.3 & \gpm{0.9} & 0 & \gpm{0} \\
\quad Navigation {\tiny(76 tasks)} & 15.8 & \gpm{1.7} & 15.8 & \gpm{2.1} & 5.3 & \gpm{1.9} & 0 & \gpm{0} \\
\bottomrule
\end{tabular}
}
\label{tab:acc-summary}
\end{table}
-----------


data:
---------------
		GPT 	se	human	se	GPT	se	human	se												
all		2.1	2	93.9	3.42	0	0	93.9	3.42												
Planning and decision making		0	0	100	0	0	0	87.5	11.69												
Information retrieval		0	0	100	0	0	0	100	0												
Data-driven decision making		0	0	84.6	10.01	0	0	100	0												
Sophisticated memory		8.3	8	91.7	7.96	0	0	91.7	7.96												
Contextual reasoning		0	0	100	0	0	0	87.5	11.69												
---------------



"""
