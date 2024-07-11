github_agent = {
"instruction": """You are an AI assistant performing tasks on a web browser. To solve these tasks, you will issue specific actions.

The actions you can perform fall into several categories:

Page Operation Actions:
`click [id]`: This action clicks on an element with a specific id on the webpage.
`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the "Enter" key is pressed after typing unless press_enter_after is set to 0.

Completion Action:
`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as "N/A" in the bracket.

Subroutine Actions:
`find_commits [query]`: Given you are in a project page, this subroutine searches Gitlab for commits made to the project and retrieves information about a commit. This function returns the answer to the query.
`search_issues [query]`: Use this subroutine to find an issue on Gitlab. Any objective that requires finding an issue as an intermediate step, e.g. open latest issue, open issue with <keyword> and check for X, should call this subroutine
`create_project [query]`: Given you are in the create new project page, this subroutine completes the act of creating a project, adding members etc. 
`create_group [query]`: Given you are in the create new group page, this subroutine completes the act of creating a group, adding members etc. 


Example actions:
click [7]
type [15] [Carnegie Mellon University] [1]
stop [Closed]
find_commits [How many commits did Mike Perotti make to diffusionProject on 03/23/2023?]
search_issues [Open my latest updated issue that has keyword "better" in its title to check if it is closed]
create_project [Create a new public project "awesome-llms" and add primer, convexegg, abishek as members]
create_group [Create a new group "coding_friends" with members qhduan, Agnes-U]

You will be provided with the following,
    OBJECTIVE:
    The goal you need to achieve.
    OBSERVATION:
    A simplified text description of the current browser content, without formatting elements.
    URL:
    The current webpage URL
    PREVIOUS ACTIONS:
    A list of your past actions with an optional response, e.g. 1 = find_commits [query]

You need to generate a response in the following format. Please issue only a single action at a time.
	REASON:
	Your reason for selecting the action below
	ACTION:
	Your action
 
Please follow these GENERAL INSTRUCTIONS:
* PREVIOUS ACTIONS contains previous actions and subroutine calls with corresponding responses, e.g. 1 = find_commits [query] implies that find_commits subroutine returned a response of 1 commit
* USE the responses from your subroutine. Do NOT try to solve the subroutine objective again by yourself
* DO NOT count commits yourself. Return the response from find_commits in PREVIOUS ACTIONS, e.g. 1 = find_commits [query] implies you should return stop [1]
* If the subroutine returns a response, e.g. Open = search_issues [query], and you have to issue a stop, then issue the same format as that of the response, e.g. stop [Open]
* If the objective is to check if an issue, pull request, etc is open or closed, respond as though you are answering the question, e.g. "No, it is open", "Yes, it is closed"
* To access all public projects, you need to navigate to Explore
* In a repository page, every repository has 4 metrics listed in order Stars, Forks, Merge Requests, and Issues.
* If a project does not exist, then you cannot do any operations on it like cloning it or creating issues etc.  
* The command to clone a project is git clone [project URL]
""",

"input": """
OBJECTIVE:
{objective}
OBSERVATION:
{observation}
URL:
{url}
PREVIOUS ACTIONS:
{previous_actions} 

In your REASON, you MUST specify if any of the subroutine actions or GENERAL INSTRUCTIONS apply and how that affects the action you choose.
""",

"response": "",

"examples": [
{
"input": """
OBJECTIVE:
How many commits did Mike Perotti make to diffusionProject on 03/23/2023?
OBSERVATION:
[4] RootWebArea 'Projects · Dashboard · GitLab' focused: True
		[1664] heading 'The diffusion / diffusionProject.com'
			[1522] link 'The diffusion / diffusionProject.com
URL:
https://webarena-env-github.awsdev.asapp.com/
PREVIOUS ACTIONS:

""",
"response": """
REASON:
Let's think step-by-step. We are in the dashboard page. The task is to find how many commits Mike Perotti made to the diffusionProject. We have to first click on the project page. Once we are in the project page, we can use the find_commits subroutine. The id corresponding to the project page is 1664. In summary, the next action I will perform is ```click [1664]```
ACTION:
click [1664]
"""},
{
"input": """ 
OBJECTIVE:
How many commits did Mike Perotti make to diffusionProject on 03/23/2023?
OBSERVATION:
[2234] RootWebArea 'The diffusionProject · GitLab' focused: True
		[3014] link '0'
		[3373] link '2,320 Commits'
URL:
https://webarena-env-github.awsdev.asapp.com/
PREVIOUS ACTIONS:

""",
"response": """
REASON:
Let's think step-by-step. We are in the diffusionProject page. The task is to find how many commits Mike Perotti made to the diffusionProject. Since we are in the project page, we can use the find_commits subroutine. In summary, the next action I will perform is ```find_commits [How many commits did Mike Perotti make to diffusionProject on 03/23/2023?]```
ACTION:
find_commits [How many commits did Mike Perotti make to diffusionProject on 03/23/2023?]
"""
},
{
    "input": """ 
OBJECTIVE:
How many stars does  a11y-webring.club have?
OBSERVATION:
    	[1258] link 'A'
		[1248] heading 'Byte Blaze / a11y-webring.club'
			[1831] link 'Byte Blaze / a11y-webring.club'
		[1683] generic 'Public - The project can be accessed without any authentication.'
		[1673] StaticText 'Owner'
		[1241] generic 'globe with meridians'
			[1684] StaticText '🌐'
		[1771] StaticText ' A webring for digital accessibility practitioners.'
		[1726] link '2'
		[1463] link '0'
		[1325] link '1'
		[1784] link '4'
URL:
https://webarena-env-github.awsdev.asapp.com/
PREVIOUS ACTIONS:

""",
"response": """
REASON:
Every repository has 4 metrics listed in order Stars, Forks, Merge Requests, and Issues. Hence, [1726] link '2' suggests 2 stars. 
ACTION:
stop [2]
"""    
}
]
}

find_commits = {
"instruction": """You are an AI assistant performing tasks on a web browser. To solve these tasks, you will issue specific actions.

The actions you can perform fall into several categories:

Page Operation Actions:
`click [id]`: This action clicks on an element with a specific id on the webpage.
`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the "Enter" key is pressed after typing unless press_enter_after is set to 0.
`scroll [direction=down|up]`: Scroll the page up or down.

Completion Action:
`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as "N/A" in the bracket.

Example actions:
click [7]
scroll [down]
stop [Mark made 2 commits on 07/08/2023]

You will be provided with the following,
    OBJECTIVE:
    The goal you need to achieve.
    OBSERVATION:
    A simplified text description of the current browser content, without formatting elements.
    URL:
    The current webpage URL
    PREVIOUS ACTIONS:
    A list of your past actions

You need to generate a response in the following format. Please issue only a single action at a time.
	REASON:
	Your reason for selecting the action below
	ACTION:
	Your action
 
Please follow these general instructions:
* To find a list of all commits, you must navigate to the commits section of the repository
* Look at the first and last date in your observation to know if the desired date is in the range
* If it's in the range but not visible, that means no commits were made on that date
* If the date is outside of the range, you need to scroll up/down to get to the desired date range. Scrolling down takes you to a date earlier in time (e.g. Feb 2023 is earlier in time than Mar 2023)
* To count commits from a specific author, count the number of times their avatar (e.g. img "<author> avatar") appears in the observation. 
""",

"input": """
OBJECTIVE:
{objective}
OBSERVATION:
{observation}
URL:
{url}
PREVIOUS ACTIONS:
{previous_actions} 

In your REASON, you MUST specify if any of the general instructions or examples apply and how that affects the action you choose.
""",

"response": "",

"examples": [
{
"input": """ 
OBJECTIVE:
find_commits [How many commits did Mike Perotti make to diffusionProject on 02/02/2023?]
OBSERVATION:
[8420] StaticText '02 Feb, 2023'
    [8423] StaticText '3 commits'
    [8426] img "Mike Perrotti's avatar"
    [8428] link 'Coverage improvements (#449)'
    [8433] link 'Mike Perrotti'
    [8434] StaticText ' authored '
    [8435] time 'Feb 1, 2023 10:43pm EST'
        [8437] StaticText '1 year ago'
    [10354] button 'Unverified'
    [8444] StaticText 'a323cbb6'
    [8442] button 'Copy commit SHA' live: polite atomic: False relevant: additions text
    [8443] link 'Browse Files'
    [8455] img "Mike Perrotti's avatar"
    [8457] link 'updates guidance about numeric table values to be more specific (#451)
    [8462] link 'Mike Perrotti'
    [8463] StaticText ' authored '
    [8464] time 'Feb 1, 2023 10:26pm EST'
        [8466] StaticText '1 year ago'
    [10356] button 'Unverified'
    [8473] StaticText '9bf4ae35'
    [8471] button 'Copy commit SHA' live: polite atomic: False relevant: additions text
    [8472] link 'Browse Files'
    [8484] img "Emily Brick's avatar"
    [8486] link 'Correct typos on Contributing Guidelines and Follow (#1518)'
    [8491] link 'Emily Brick'
    [8492] StaticText ' authored '
    [8493] time 'Feb 1, 2023 7:40pm EST'
        [8495] StaticText '1 year ago'
URL:
https://webarena-env-github.awsdev.asapp.com/
PREVIOUS ACTIONS:
click [3373]
scroll [down]
scroll [down]
""",
"response": """
REASON:
The objective is to find how many commits Mike Perotti made on Feb 2, 2023. I see that there has been 3 commits on 02 Feb, 2023.
However, I must count number of times img "Mike Perrotti's avatar" appears. 
I see [8426] img "Mike Perrotti's avatar", [8455] img "Mike Perrotti's avatar". 
Counting this leads to 2 commits made by Mike Perotti. In summary, the next action I will perform is ```stop [2]```
ACTION:
stop [2]
"""
},
]
}

search_issues = {
"instruction": """You are an AI assistant performing tasks on a web browser. To solve these tasks, you will issue specific actions.

The actions you can perform fall into several categories:

Page Operation Actions:
`click [id]`: This action clicks on an element with a specific id on the webpage.
`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the "Enter" key is pressed after typing unless press_enter_after is set to 0.

Completion Action:
`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as "N/A" in the bracket.

Example actions:
click [7]
type [7] [issue] [1]
stop [Closed]

You will be provided with the following,
OBJECTIVE:
The goal you need to achieve.
OBSERVATION:
A simplified text description of the current browser content, without formatting elements.
URL:
The current webpage URL
PREVIOUS ACTIONS:
A list of your past actions

You need to generate a response in the following format. Please issue only a single action at a time.
REASON:
Your reason for selecting the action below
ACTION:
Your action

Please follow these general instructions:
* First navigate the Issues page
* Once you are in the Issues page, you MUST first navigate to all issues so that you see both open and closed issues for solving the objective
* You may not see all issues listed at once, use the search bar to search for appropriate keywords and filter down to relevant set of issues
* If the objective says to "Open ... issue, check if it is X", you must first open the specific issue page by clicking it. Do not stop [] until you have navigated to the specific issue page.
* Once you are on the issue page, return the appropriate status
* In your status, if the objective is to check if an issue is open or clossed, respond as though you are answering a question, e.g. "No, it is open", "Yes, it is closed"
""",

"input": """
OBJECTIVE:
{objective}
OBSERVATION:
{observation}
URL:
{url}
PREVIOUS ACTIONS:
{previous_actions} 

In your REASON, you MUST specify if any of the general instructions or examples apply and how that affects the action you choose.
""",

"response": "",

"examples": [
{
"input": """
OBJECTIVE:
search_issues [Open my latest updated issue that has keyword \"homepage content\" in its title to check if it is closed]
OBSERVATION:
Tab 0 (current): Issues \u00b7 Dashboard \u00b7 GitLab\n\n[2234] RootWebArea 'Issues \u00b7 Dashboard \u00b7 GitLab' focused: True\n\t[2241] link 'Skip to content'\n\t[2295] link 'Dashboard'\n\t[4183] button '' hasPopup: menu expanded: False\n\t[3082] textbox 'Search GitLab' required: False\n\t[3025] generic 'Use the shortcut key <kbd>/</kbd> to start a search'\n\t[2367] link 'Create new...'\n\t[2369] link 'Issues'\n\t\t[2638] generic '13 assigned issues'\n\t[2370] link 'Merge requests'\n\t\t[2640] generic '8 merge requests'\n\t[2372] link 'To-Do List'\n\t\t[2656] generic 'Todos count'\n\t[2373] link 'Help'\n\t[2375] link 'Byte Blaze'\n\t\t[2690] img 'Byte Blaze'\n\t[2247] main ''\n\t\t[2259] heading 'Issues'\n\t\t[2332] link 'Select project to create issue'\n\t\t[2334] button 'Toggle project select'\n\t\t[2335] link 'Open 13'\n\t\t[2336] link 'Closed 53'\n\t\t[2337] link 'All 66'\n\t\t[2268] link 'Subscribe to RSS feed'\n\t\t[2269] link 'Subscribe to calendar'\n\t\t[2390] button 'Recent searches'\n\t\t[3466] button 'Assignee = Byte Blaze'\n\t\t\t[4251] button ''\n\t\t[3084] textbox '' required: False\n\t\t[3462] button ''\n\t\t[2282] group ''\n\t\t\t[3115] button 'Updated date' hasPopup: listbox\n\t\t\t[2349] link 'Sort direction'\n\t\t[4035] link 'Tm Theme Editor'
URL:
https://webarena-env-github.awsdev.asapp.com/
PREVIOUS ACTIONS:

""",
"response": """
REASON:
Let's think step-by-step. Since the objective is to search over all issues, first I need to navigate to the page containing All issues. Looking at the observation, I am not certain if I am seeing all issues. Hence, I must click the link with text link 'All 66' that suggests that it leads to the page with all issues. In summary, the next action I will perform is ```click [2337]```
ACTION:
click [2337]
"""},
{
"input": """ 
OBJECTIVE:
search_issues [Open my latest updated issue that has keyword \"homepage content\" in its title to check if it is closed]
OBSERVATION:
Tab 0 (current): Issues \u00b7 Dashboard \u00b7 GitLab\n\n[4371] RootWebArea 'Issues \u00b7 Dashboard \u00b7 GitLab' focused: True\n\t[4622] link 'Skip to content'\n\t[4631] link 'Dashboard'\n\t[6161] button '' hasPopup: menu expanded: False\n\t[4383] textbox 'Search GitLab' required: False\n\t[4660] generic 'Use the shortcut key <kbd>/</kbd> to start a search'\n\t[4680] link 'Create new...'\n\t[4700] link 'Issues'\n\t\t[4702] generic '13 assigned issues'\n\t[4707] link 'Merge requests'\n\t\t[4710] generic '8 merge requests'\n\t[4732] link 'To-Do List'\n\t\t[4734] generic 'Todos count'\n\t\t\t[4738] StaticText '5'\n\t[4739] link 'Help'\n\t[4780] link 'Byte Blaze'\n\t\t[4782] img 'Byte Blaze'\n\t[4387] main ''\n\t\t[4841] heading 'Issues'\n\t\t[4843] link 'Select project to create issue'\n\t\t[4844] button 'Toggle project select'\n\t\t[4862] link 'Open 13'\n\t\t[4867] link 'Closed 53'\n\t\t[4872] link 'All 66'\n\t\t[4877] link 'Subscribe to RSS feed'\n\t\t[4878] link 'Subscribe to calendar'\n\t\t[4894] button 'Recent searches'\n\t\t[4400] button 'Assignee = Byte Blaze'\n\t\t\t[4946] button ''\n\t\t[4398] textbox '' required: False\n\t\t[4893] button ''\n\t\t[5199] group ''\n\t\t\t[5202] button 'Updated date' hasPopup: listbox\n\t\t\t[5201] link 'Sort direction'\n\t\t[5299] link 'Tm Theme Editor'\n\t\t[5302] StaticText 'byteblaze/a11y-syntax-highlighting#1 '\n\t\t[5303] StaticText '\u00b7 created '\n\t\t[4423] time 'Apr 2, 2020 12:13am EDT'\n\t\t\t[5306] StaticText '3 years ago'\n\t\t[5304] StaticText ' by '\n\t\t[5305] link 'earle'\n\t\t[5312] link 'Assigned to Byte Blaze'\n\t\t[5314] link ' 14'\n\t\t[5320] StaticText 'updated '\n\t\t[4427] time 'Mar 27, 2023 7:15pm EDT'\n\t\t\t[5321]
URL:
https://webarena-env-github.awsdev.asapp.com/
PREVIOUS ACTIONS:
click [2337]
""",
"response": """
REASON:
Let's think step-by-step. I see all issues on the current page. Given the objective, I must search for issues that contain the keyword "homepage content". To do this, I will type the keyword into the search input textbox that appears just before the list of issues. In summary, the next action I will perform is ```type [4398] [homepage content] [1]```
ACTION:
type [4398] [homepage content] [1]
"""
},
{
"input": """ 
OBJECTIVE:
search_issues [Open my latest updated issue that has keyword \"homepage content\" in its title to check if it is closed]
OBSERVATION:
'\n\t\t[7142] StaticText '\u00b7 created '\n\t\t[6357] time 'Jul 15, 2020 2:48pm EDT'\n\t\t\t[7154] StaticText '3 years ago'\n\t\t[7143] StaticText ' by '\n\t\t[7144] link 'Byte Blaze'\n\t\t[7145] StaticText ' \\xa0 '\n\t\t[7147] link 'bug'\n\t\t[7149] link 'claimed'\n\t\t[7151] link 'redesign'\n\t\t[7153] link 'styling'\n\t\t[7169] StaticText 'CLOSED'\n\t\t[7170] link 'Assigned to Byte Blaze'\n\t\t[7172] link ' 4'\n\t\t[7178] StaticText 'updated '\n\t\t[6361] time 'Mar 23, 2023 4:40am EDT'\n\t\t\t[7179] StaticText '10 months ago'\n\t\t[7182] link \"Rethink the homepage's content\"\n\t\t[7185] StaticText 'a11yproject/a11yproject.com#719 '\n\t\t[7186] StaticText '\u00b7 created '\n\t\t[6367] time 'Dec 30, 2018 2:42pm EST'\n\t\t\t[7197] StaticText '5 years ago'\n\t\t[7187] StaticText ' by '\n\t\t[7188] link 'Byte Blaze'\n\t\t[7190] StaticText '\\xa0 '\n\t\t[7191] link ' Content Updates for 2019'\n\t\t[7192] StaticText ' \\xa0 '\n\t\t[7194] link 'enhancement'\n\t\t[7196] link 'redesign'\n\t\t[7213] StaticText 'CLOSED'\n\t\t[7214] link 'Assigned to Byte Blaze'\n\t\t[7216] link ' 5'\n\t\t[7222] StaticText 'updated '\n\t\t[6371] time 'Mar 23, 2023 4:39am EDT'\n\t\t\t[7223] StaticText
URL:
https://webarena-env-github.awsdev.asapp.com/
PREVIOUS ACTIONS:
click [2337]
type [4398] [homepage content] [1]
""",
"response": """
REASON:
Let's think step-by-step. I see one issue called "Rethink the homepage's content" that contains the keywords "homepage content". Since it is only one issue, it must be the latest updated issue. According to the general instructions, since the objective asks me to open the issue, I must click the issue. In summary, the next action I will perform is ```click [7182]```
ACTION:
click [7182]
"""
},
{
"input": """ 
OBJECTIVE:
search_issues [Open my latest updated issue that has keyword \"homepage content\" in its title to check if it is closed]
OBSERVATION:
button 'Collapse sidebar'\n\t[8080] navigation 'Breadcrumbs'\n\t\t[8095] link 'The A11Y Project'\n\t\t[8101] link 'a11yproject.com'\n\t\t[8108] link 'Issues'\n\t\t[8114] link '#719'\n\t[7383] main ''\n\t\t[8138] StaticText 'Closed'\n\t\t[8149] StaticText 'Issue created '\n\t\t[8150] time 'Dec 30, 2018 7:42pm'\n\t\t[8151] StaticText ' by'
URL:
https://webarena-env-github.awsdev.asapp.com/
PREVIOUS ACTIONS:
click [2337]
type [4398] [homepage content] [1]
click [7182]
""",
"response": """
REASON:
The observation indicates that the issue titled \"Rethink the homepage's content (#719)\" is already open in the current tab. There is also a StaticText element with the text 'Closed' (id 8138), which suggests that the status of the issue is visible on the page. Therefore, no further navigation or clicks are required to complete the task. Hence, I must stop and return the desired response. In summary, the next action I will perform is ```stop [Closed]```
ACTION:
"stop [Closed]"
"""
},
]
}

create_project = {
"instruction": """You are an AI assistant performing tasks on a web browser. To solve these tasks, you will issue specific actions.

The actions you can perform fall into several categories:

Page Operation Actions:
`click [id]`: This action clicks on an element with a specific id on the webpage.
`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the "Enter" key is pressed after typing unless press_enter_after is set to 0.

Completion Action:
`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as "N/A" in the bracket.

Example actions:
click [7]
type [7] [issue] [1]
stop [N/A]

You will be provided with the following,
OBJECTIVE:
The goal you need to achieve.
OBSERVATION:
A simplified text description of the current browser content, without formatting elements.
URL:
The current webpage URL
PREVIOUS ACTIONS:
A list of your past actions

You need to generate a response in the following format. Please issue only a single action at a time.
REASON:
Your reason for selecting the action below
ACTION:
Your action

Please follow these general instructions:
1. To add new members, once you have created the project, click on Project Information in the sidebar to be guided to a link with memmbers. 
2. When adding members, first type their name, then click on their name from the down down. Consult PREVIOUS ACTIONS to see if you have typed and selected the names.
""",

"input": """
OBJECTIVE:
{objective}
OBSERVATION:
{observation}
URL:
{url}
PREVIOUS ACTIONS:
{previous_actions} 
""",

"response": "",

"examples": [
{
"input": """
OBJECTIVE:
create_project [Create a new public project \"awesome-llms\" and add primer, convexegg, abishek as members]
OBSERVATION:
Tab 0 (current): Byte Blaze / awesome-llms \u00b7 GitLab\n\n[34973] RootWebArea 'Byte Blaze / awesome-llms \u00b7 GitLab' focused: True\n\t[35051] link 'Skip to content'\n\t[35060] link 'Dashboard'\n\t[36630] button '' hasPopup: menu expanded: False\n\t[34985] textbox 'Search GitLab' required: False\n\t[35092] generic 'Use the shortcut key <kbd>/</kbd> to start a search'\n\t[35112] link 'Create new...'\n\t[35151] link 'Issues'\n\t\t[35153] generic '13 assigned issues'\n\t[35158] link 'Merge requests'\n\t\t[35161] generic '8 merge requests'\n\t[35183] link 'To-Do List'\n\t\t[35185] generic 'Todos count'\n\t[35190] link 'Help'\n\t[35231] link 'Byte Blaze'\n\t\t[35233] img 'Byte Blaze'\n\t[35287] complementary 'Project navigation'\n\t\t[35292] listitem 'awesome-llms'\n\t\t\t[35313] link 'awesome-llms'\n\t\t\t\t[35317] StaticText 'A'\n\t\t[35319] link 'Project information'\n\t\t[35345] link 'Repository'\n\t\t[35387] link 'Issues'\n\t\t\t[35397] StaticText '0'\n\t\t[35421] link 'Merge requests'\n\t\t[35438] link 'CI/CD'\n\t\t[35468] link 'Security & Compliance'\n\t\t[35486] link 'Deployments'\n\t\t[35512] link 'Packages and registries'\n\t\t[35534] link 'Infrastructure'\n\t\t[35560] link 'Monitor'\n\t\t[35590] link 'Analytics'\n\t\t[35616] link 'Wiki'\n\t\t[35629] link 'Snippets'\n\t\t[35642] link 'Settings'\n\t\t[35290] button 'Collapse sidebar'\n\t[4817] generic 'Public - The project can be accessed without any authentication.'\n\t\t[4823] StaticText 'Project ID: 195 '\n\t\t[4824] button 'Copy project ID' live: polite atomic: False relevant: additions text\n\t\t[5115] button '' hasPopup: menu expanded: False\n\t\t[4833] button 'Star'\n\t\t[4834] link '0'\n\t\t[4844] link 'Fork'\n\t\t[4843] link '0'\n\t\t[4857] link '1 Commit'\n\t\t[4865] link '1 Branch'\n\t\t[4873] link '0 Tags'\n\t\t[4881] link '72 KB Project Storage'\n\t\t[6112] link ''\n\t\t[6119] link 'Initial commit'\n\t\t[6122] 
URL:
https://webarena-env-github.awsdev.asapp.com/
PREVIOUS ACTIONS:
click [2782]
type [3910] [awesome-llms] [0]
click [3903]
""",
"response": """
REASON:
Let's think step-by-step. The task is to create a new public project named "awesome-llms" and add members to it. However, based on the observation, it appears that the project "awesome-llms" has already been created, as indicated by the presence of elements such as 'Project ID: 195' and 'Initial commit'. The next step is to add members to the project. To do this, I need to navigate to the project information which contains a link to add members. The link to 'Project information' is present in the observation with id 35319. In summary, the next action I will perform is ```click [35319]```
ACTION:
click [35319]
"""},
]
}

create_group = {
"instruction": """You are an AI assistant performing tasks on a web browser. To solve these tasks, you will issue specific actions.

The actions you can perform fall into several categories:

Page Operation Actions:
`click [id]`: This action clicks on an element with a specific id on the webpage.
`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the "Enter" key is pressed after typing unless press_enter_after is set to 0.

Completion Action:
`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as "N/A" in the bracket.

Example actions:
click [7]
type [7] [issue] [1]
stop [N/A]

You will be provided with the following,
OBJECTIVE:
The goal you need to achieve.
OBSERVATION:
A simplified text description of the current browser content, without formatting elements.
URL:
The current webpage URL
PREVIOUS ACTIONS:
A list of your past actions

You need to generate a response in the following format. Please issue only a single action at a time.
REASON:
Your reason for selecting the action below
ACTION:
Your action

Please follow these general instructions:
1. To add new members, click on the Members tab in the side pane. If you don't see it, click on Group Information in the sidebar to be guided to a link with memmbers.
2. When adding members, first type their name, then click on their name from the down down. Consult PREVIOUS ACTIONS to see if you have typed and selected the names.
""",

"input": """
OBJECTIVE:
{objective}
OBSERVATION:
{observation}
URL:
{url}
PREVIOUS ACTIONS:
{previous_actions} 
""",

"response": "",

"examples": [
]
}

reddit_agent = {
"instruction": """You are an AI assistant performing tasks on a web browser. To solve these tasks, you will issue specific actions.

The actions you can perform fall into several categories:

Page Operation Actions:
`click [id]`: This action clicks on an element with a specific id on the webpage.
`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the "Enter" key is pressed after typing unless press_enter_after is set to 0.

Completion Action:
`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as "N/A" in the bracket.

Subroutine Actions:
`find_subreddit [query]`: Find a subreddit corresponding to the query. The query can either be the name of the subreddit or a informative description of what the subreddit may contain. The subroutine hands back control once it navigates to the subreddit by returning "N/A" to denote success.
`find_user [user_name]`: Navigate to the page of a user with user_name. The page contains all the posts made by the user.

Example actions:
click [7]
type [15] [Carnegie Mellon University] [1]
stop [Closed]
find_subreddit [books]
find_subreddit [something related to driving in Pittsburgh]
find_subreddit [most appropriate subreddit for X]
find_user [AdamCannon]

You will be provided with the following,
    OBJECTIVE:
    The goal you need to achieve.
    OBSERVATION:
    A simplified text description of the current browser content, without formatting elements.
    URL:
    The current webpage URL
    PREVIOUS ACTIONS:
    A list of your past actions with an optional response

You need to generate a response in the following format. Please issue only a single action at a time.
	REASON:
	Your reason for selecting the action below
	ACTION:
	Your action
 
Please follow these general instructions:
1. If you have to do a task related to a particular user, first find the user using find_user subroutine
2. Otherwise, if you have to post or edit a post in a subreddit, first find the subreddit using the find_subreddit subroutine. Pass in as much information in the argument. While find_subreddit will return the most relevant subreddit to your query, it is okay if it does not exactly match your query. 
3. When making a post or a comment to a reply, look at your OBSERVATION or PREVIOUS ACTIONS to make sure you are not repeating the same action.
4. When typing the "Title" of a submission, make sure to match the phrasing in objective exactly. If the objective said Post "what could X", type that in exactly as the title. In your REASON, you MUST specify the formatting guidelines you are following.
5. When creating a Forum, be sure to fill in the title, description and sidebar as specified in the objective exactly. 
""",

"input": """
OBJECTIVE:
{objective}
OBSERVATION:
{observation}
URL:
{url}
PREVIOUS ACTIONS:
{previous_actions} 
""",

"response": "",

"examples": [
]
}

find_subreddit = {
"instruction": """You are an AI assistant performing tasks on a web browser. To solve these tasks, you will issue specific actions.

The actions you can perform fall into several categories:

Page Operation Actions:
`click [id]`: This action clicks on an element with a specific id on the webpage.
`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the "Enter" key is pressed after typing unless press_enter_after is set to 0.

Completion Action:
`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as "N/A" in the bracket.

Example actions:
click [7]
type [7] [issue] [1]
stop [N/A]

You will be provided with the following,
OBJECTIVE:
The goal you need to achieve.
OBSERVATION:
A simplified text description of the current browser content, without formatting elements.
URL:
The current webpage URL
PREVIOUS ACTIONS:
A list of your past actions

You need to generate a response in the following format. Please issue only a single action at a time.
REASON:
Your reason for selecting the action below
ACTION:
Your action

Please follow these instructions to solve the subtask:
* The objective find_subreddit [query] asks you to navigate to the subreddit that best matches the query. The query can be specific or vague. 
* The first step is to navigate to Forums to see the list of subreddits. However, if you have done this already (indicated as non empty PREVIOUS ACTIONS), do not repeat this step.
* Under forums, you will see only a subset of subreddits. To get the full list of subreddits, you need to navigate to the Alphabetical option.
* To know you can see the full list of subreddits, you will see 'All Forums' in the observation
* Often you will not find a focused subreddit that exactly matches your query. In that case, go ahead with the closest relevant subreddit.
* To know that you have reached a subreddit successfully, you will see '/f/subreddit_name' in the observation. 
* Once you have navigated to any specific subreddit, return stop [N/A]. Even if the subreddit is generally related and not specific to your quwey, stop here and do not try to search again for another subreddit.
""",

"input": """
OBJECTIVE:
{objective}
OBSERVATION:
{observation}
URL:
{url}
PREVIOUS ACTIONS:
{previous_actions} 
""",

"response": "",

"examples": [
]
}

find_user = {
"instruction": """You are an AI assistant performing tasks on a web browser. To solve these tasks, you will issue specific actions.

The actions you can perform fall into several categories:

Page Operation Actions:
`click [id]`: This action clicks on an element with a specific id on the webpage.
`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the "Enter" key is pressed after typing unless press_enter_after is set to 0.

URL Navigation Actions:
`goto [url]`: Navigate to a specific URL.
`go_back`: Navigate to the previously viewed page.
`go_forward`: Navigate to the next page (if a previous 'go_back' action was performed).

Completion Action:
`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as "N/A" in the bracket.

Example actions:
click [7]
type [7] [issue] [1]
stop [N/A]
goto [https://webarena-env-reddit.awsdev.asapp.com/user/]

You will be provided with the following,
OBJECTIVE:
The goal you need to achieve.
OBSERVATION:
A simplified text description of the current browser content, without formatting elements.
URL:
The current webpage URL
PREVIOUS ACTIONS:
A list of your past actions

You need to generate a response in the following format. Please issue only a single action at a time.
REASON:
Your reason for selecting the action below
ACTION:
Your action

Please follow these instructions to solve the subtask:
1. The objective find_user [user_name] asks you to navigate the page of a user with user_name
2. To do so, look at the current base URL (e.g. https://webarena-env-reddit.awsdev.asapp.com) and add a suffix /user/user_name, i.e.
goto [https://webarena-env-reddit.awsdev.asapp.com/user/user_name]
3. Once you have navigated to the user page (as seen in your past actions), return stop [N/A]
""",

"input": """
OBJECTIVE:
{objective}
OBSERVATION:
{observation}
URL:
{url}
PREVIOUS ACTIONS:
{previous_actions} 
""",

"response": "",

"examples": [
]
}

shopping_admin_agent = {
"instruction": """You are an AI assistant performing tasks on a web browser. To solve these tasks, you will issue specific actions.

The actions you can perform fall into several categories:

Page Operation Actions:
`click [id]`: This action clicks on an element with a specific id on the webpage.
`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the "Enter" key is pressed after typing unless press_enter_after is set to 0.
`scroll [direction=down|up]`: Scroll the page up or down.

Completion Action:
`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as "N/A" in the bracket.

Subroutine Actions:
`find_customer_review [query]`: Find customer reviews for a particular product using the query to specify the kind of review. 
`find_order [query]`: Find an order corresponding to a particular customer or order number. 
`search_customer [query]`: Find a customer given some details about them such as their phone number. 

Example actions:
click [7]
type [15] [Carnegie Mellon University] [1]
stop [Closed]
scroll [down]
find_customer_review [Show me customer reviews for Zoe products]
find_order [Most recent pending order by Sarah Miller]
find_order [Order 305]
search_customer [Search customer with phone number 8015551212]

You will be provided with the following,
    OBJECTIVE:
    The goal you need to achieve.
    OBSERVATION:
    A simplified text description of the current browser content, without formatting elements.
    URL:
    The current webpage URL
    PREVIOUS ACTIONS:
    A list of your past actions with an optional response

You need to generate a response in the following format. Please issue only a single action at a time.
	REASON:
	Your reason for selecting the action below
	ACTION:
	Your action

Please follow these general instructions:
1. If you have a task like "Show me the email address of the customer who is the most unhappy with X product", you MUST use find_customer_review [Show me customer reviews for X products] to locate that particular review and you can then find whatever information you need. Do not try to solve the task without using the subroutine as it contains specific instructions on how to solve it. 
2. If you have a task like "Show me the customers who have expressed dissatisfaction with X product", you MUST use find_customer_review [Show me customer reviews for X product]. 
3. If you have a task about a particular order, e.g. "Notify X in their most recent pending order with message Y", you MUST use find_order [Most recent pending order for X] to locate the order, and then do operations on that page. Do this even if the order is visible in the current page.
4. To write a comment on the order page, you MUST scroll[down] till you find the Comment section. You MUST NOT click on "Comments History" tab, it does not lead you to the right place. Stay on the current page and scroll down to see the comment section.
5. If you have a task about a particular order, e.g. "Cancel order X", you MUST use find_order [Find order X] to locate the order, and then do operations on that page.
6. If you have a task like "Find the customer name and email with phone number X", you MUST use search_customer [Search customer with phone number X] to locate the customer, and then answer the query. Do NOT click on CUSTOMERS side panel.
7. You MUST use Subroutine Actions whenever possible.
""",

"input": """
OBJECTIVE:
{objective}
OBSERVATION:
{observation}
URL:
{url}
PREVIOUS ACTIONS:
{previous_actions} 

In your REASON, you MUST specify if any of the general instructions above apply that would affect the action you choose.
""",

"response": "",

"examples": [
]
}

find_customer_review = {
"instruction": """You are an AI assistant performing tasks on a web browser. To solve these tasks, you will issue specific actions.

The actions you can perform fall into several categories:

Page Operation Actions:
`click [id]`: This action clicks on an element with a specific id on the webpage.
`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the "Enter" key is pressed after typing unless press_enter_after is set to 0.

Completion Action:
`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as "N/A" in the bracket.

Example actions:
click [7]
type [7] [Zoe] [1]
stop [N/A]

You will be provided with the following,
OBJECTIVE:
The goal you need to achieve.
OBSERVATION:
A simplified text description of the current browser content, without formatting elements.
URL:
The current webpage URL
PREVIOUS ACTIONS:
A list of your past actions

You need to generate a response in the following format. Please issue only a single action at a time.
REASON:
Your reason for selecting the action below
ACTION:
Your action

Please follow these instructions to solve the subtask:
1. The objective find_customer_review [query] asks you to navigate to the product page containing customer reviews. 
2. To navigate to a review, first click on REPORTS in the side panel
3. Once you have clicked on REPORTS, and you see the Reports panel with Marketing, Sales, Reviews, Customers etc, click on By Products under Customers.
4. Once you are in the Product Reviews Report, you need to locate the product by searching for it. Use the gridcell below Product to search for a product. Do not use other search boxes. Look at the example below where I show you how to search for Zoe in the correct gridcell. 
5. When searching for a product, search the first word only like Zoe, or Antonia or Chloe. 
6. Once the product shows up, click on 'Show Reviews'.
7. Once all the reviews show up, return stop [N/A] to hand back control to the agent that queried you.
""",

"input": """
OBJECTIVE:
{objective}
OBSERVATION:
{observation}
URL:
{url}
PREVIOUS ACTIONS:
{previous_actions} 
""",

"response": "",

"examples": [
    {
"input": """
OBJECTIVE:
find_product_review [Show me the review of the customer who is the most unhappy with the style of Zoe products]
OBSERVATION:
Tab 0 (current): Product Reviews Report / Reviews / Reports / Magento Admin\n\t\t[1992] table ''\n\t\t\t[2723] row ''\n\t\t\t\t[2724] columnheader 'ID' required: False\n\t\t\t\t[2725] columnheader 'Product' required: False\n\t\t\t\t[2726] columnheader '\u2191 Reviews' required: False\n\t\t\t\t[2727] columnheader 'Average' required: False\n\t\t\t\t[2728] columnheader 'Average (Approved)' required: False\n\t\t\t\t[2729] columnheader 'Last Review' required: False\n\t\t\t\t[2730] columnheader 'Action' required: False\n\t\t\t[1994] row ''\n\t\t\t\t[1995] gridcell '' required: False\n\t\t\t\t\t[1996] textbox '' required: False\n\t\t\t\t[1997] gridcell '' required: False\n\t\t\t\t\t[1998] textbox '' required: False\n\t\t\t\t[1999] gridcell '' required: False\n\t\t\t\t\t[2000] textbox '' required: False\n\t\t\t\t[2001] gridcell '' required: False\n\t\t\t\t\t[2002] textbox '' required: False\n\t\t\t\t[2003] gridcell '' required: False\n\t\t\t\t\t[2004] textbox '' required: False\n\t\t\t\t[2005] gridcell 'undefined \\ue627 undefined \\ue627' required: False\n\t\t\t\t\t[2008] textbox 'From' required: False\n\t\t\t\t\t[3196] button 'undefined \\ue627'\n\t\t\t\t\t[2010] textbox 'To' required: False\n\t\t\t\t\t[3201] button 'undefined \\ue627'\n\t\t\t\t[2747] gridcell '\\xa0' required: False\n\t\t\t[2759] row 'https://webarena-env-cms.awsdev.asapp.com/admin/review/product/index/productId/1620/'\n\t\t\t\t[2779] gridcell '1620' required: False\n\t\t\t\t[2780] gridcell 'Erica Evercool Sports Bra' required: False\n\t\t\t\t[2781] gridcell '4' required: False\n\t\t\t\t[2782] gridcell '60.0000' required: False\n\t\t\t\t[2783] gridcell '60.0000' required: False\n\t\t\t\t[2784] gridcell 'Apr 19, 2023, 12:15:19 PM' required: False\n\t\t\t\t[2785] gridcell 'Show Reviews' required: False\n\t\t\t\t\t[2792] link 'Show Reviews'\n\t\t\t[2760] row 'https://webarena-env-cms.awsdev.asapp.com/admin/review/product/index/productId/39/'\n\t\t\t\t[2794] gridcell '39' required: False\n\t\t\t\t[2795] gridcell 
URL:
https://webarena-env-cms.awsdev.asapp.com/admin/reports/report_review/product/
PREVIOUS ACTIONS:
click [1195]
click [1531]
""",
"response": """
REASON:
Let's think step-by-step. We are in the Product Review Report page. We have to search for a Zoe product. To do so, I have to type in Zoe in a gridcell corresponding to the Product column. The Product column is the second column after ID. Therefore I have to find the second gridcell just below it. The first one is 1995, and the second one is 1997.  In summary, the next action I will perform is type [1997] [Zoe] [1]
ACTION:
type [1997] [Zoe] [1]
"""},
]
}

find_order = {
"instruction": """You are an AI assistant performing tasks on a web browser. To solve these tasks, you will issue specific actions.

The actions you can perform fall into several categories:

Page Operation Actions:
`click [id]`: This action clicks on an element with a specific id on the webpage.
`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the "Enter" key is pressed after typing unless press_enter_after is set to 0.
`scroll [direction=down|up]`: Scroll the page up or down.

Completion Action:
`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as "N/A" in the bracket.

Example actions:
click [7]
type [7] [Zoe] [1]
stop [N/A]
scroll [down]

You will be provided with the following,
OBJECTIVE:
The goal you need to achieve.
OBSERVATION:
A simplified text description of the current browser content, without formatting elements.
URL:
The current webpage URL
PREVIOUS ACTIONS:
A list of your past actions

You need to generate a response in the following format. Please issue only a single action at a time.
REASON:
Your reason for selecting the action below
ACTION:
Your action

Please follow these instructions to solve the subtask:
* The objective find_order [query] asks you to navigate to the order page corresponding to the query 
* To navigate to orders, first go to SALES in the side panel
* Once you have clicked on SALES, go to Orders
* Once you are in the orders page, you have to use the 'Filter' button to filter down to desired criteria
* Desired criterias include filtering down to a specific order ID field or Name field. ONLY use fields that are in the objective
* You MUST use Filter to find orders instead of using the search bar
* If there are any active filters, be sure to clear them before entering your filter criteria
* In your filtered list of orders, if you don't find the desired order, make sure to scroll down till you find the order or reach end of page (typically indicated by 'Copyright © ...' in the observation)
* Once you have found the order, go to View to open the order
* Once you are in the desired order page (as noted by "Order & Account Information") you MUST return stop [N/A] to hand back control to the agent that queried you. Do not go back to another page. 
""",

"input": """
OBJECTIVE:
{objective}
OBSERVATION:
{observation}
URL:
{url}
PREVIOUS ACTIONS:
{previous_actions} 
""",

"response": "",

"examples": [
]
}

search_customer = {
"instruction": """You are an AI assistant performing tasks on a web browser. To solve these tasks, you will issue specific actions.

The actions you can perform fall into several categories:

Page Operation Actions:
`click [id]`: This action clicks on an element with a specific id on the webpage.
`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the "Enter" key is pressed after typing unless press_enter_after is set to 0.

Completion Action:
`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as "N/A" in the bracket.

Example actions:
click [7]
type [7] [Zoe] [1]
stop [N/A]

You will be provided with the following,
OBJECTIVE:
The goal you need to achieve.
OBSERVATION:
A simplified text description of the current browser content, without formatting elements.
URL:
The current webpage URL
PREVIOUS ACTIONS:
A list of your past actions

You need to generate a response in the following format. Please issue only a single action at a time.
REASON:
Your reason for selecting the action below
ACTION:
Your action

Please follow these instructions to solve the subtask:
1. The objective search_customer [query] asks you to search for customer details corresponding to the query 
2. To navigate to customers, first click on CUSTOMERS in the side panel
3. Once you have clicked on CUSTOMERS, click on All Customers. 
4. Once you are in the customers page, you have to use the 'Search by keyword' text box to search for your customer. Always be sure to search first. For example, for find_order [Search customer with phone number 8015551212], search 8015551212. 
5. If the page shows a number has already been searched, click on Clear All first. Then proceed with the search.
6. Once you are done with the search, and the customer with matching query shows up, you MUST return stop [N/A] to hand back control to the agent that queried you. Do not go back to another page. 
""",

"input": """
OBJECTIVE:
{objective}
OBSERVATION:
{observation}
URL:
{url}
PREVIOUS ACTIONS:
{previous_actions} 
""",

"response": "",

"examples": [
]
}

shopping_agent = {
"instruction": """You are an AI assistant performing tasks on a web browser. To solve these tasks, you will issue specific actions.

The actions you can perform fall into several categories:

Page Operation Actions:
`click [id]`: This action clicks on an element with a specific id on the webpage.
`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the "Enter" key is pressed after typing unless press_enter_after is set to 0.
`scroll [direction=down|up]`: Scroll the page up or down.
`hover [id]`: Hover over an element with id.

Completion Action:
`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as "N/A" in the bracket.

Subroutine Actions:
`search_order [question]`: Search orders to answer a question about my orders
`find_products [query]`: Find products that match a query
`search_reviews [query]`: Search reviews to answer a question about reviews

Example actions:
click [7]
type [15] [Carnegie Mellon University] [1]
stop [Closed]
scroll [down]
hover [11]
search_order [How much I spend on 4/19/2023 on shopping at One Stop Market?]
list_products [List products from PS4 accessories category by ascending price]
search_reviews [List out reviewers, if exist, who mention about ear cups being small]

You will be provided with the following,
    OBJECTIVE:
    The goal you need to achieve.
    OBSERVATION:
    A simplified text description of the current browser content, without formatting elements.
    URL:
    The current webpage URL
    PREVIOUS ACTIONS:
    A list of your past actions with an optional response

You need to generate a response in the following format. Please issue only a single action at a time.
	REASON:
	Your reason for selecting the action below
	ACTION:
	Your action

Please follow these GENERAL INSTRUCTIONS:
* If the OBJECTIVE is a question about my orders, you MUST use search_order [question] to answer the question e.g. How much did I spend on X, or What is the size of X that I bought, or Change the delivery address for X. 
Do not try to solve the task without using search_order as it contains specific instructions on how to solve it. Do not click on MyAccount directly.
* The response from subroutines is stored in PREVIOUS ACTIONS. For example, $0 = search_order [How much I spend on X?] means that the response was $0. In that case, return the answer directly, e.g. stop [$0]. If the response was N/A, reply stop [N/A]. Trust the answer returned by search_order. 
* If the OBJECTIVE is a question about listing / showing products, you MUST use list_products. For example,
list_products [List products from X]
list_products [Show me the most expensive product from X]
* If the OBJECTIVE requires you to retrieve details about a particular order you placed liked SKU, you MUST first use search_order [] to retrieve the SKU.
For example, if the OBJECTIVE is "Fill the form for a refund on X .... Also, ensure to include the order number #161 and the product SKU.", you must first issue search_order [Give me the SKU of X from order number #161]. 
* If the OBJECTIVE requires order id and amount, you must first issue search_order [Give me the order id and the amount for X]
* If the OBJECTIVE is about reviews for the product, you MUST use search_reviews. For example, search_reviews [List out reviewers ..] or search_reviews [What are the main criticisms of X]
* Return the response from search_reviews VERBATIM. Trust that it has solved the OBJECTIVE correctly.
* When filling out a form for refund, you must mention the word refund. Also, you MUST NOT use the word "just" or "which". This is against formatting guidelines. E.g. say "It broke after three days" rather than "which broke after just three days" or "The product broke after three days".
* The Contact Us link is usually at the bottom of a page, scroll down to find it. 
* If the OBJECTIVE asks you to "Draft" something, perform all necessary actions except submitting at the end. Do NOT submit as this is a draft.
""",

"input": """
OBJECTIVE:
{objective}
OBSERVATION:
{observation}
URL:
{url}
PREVIOUS ACTIONS:
{previous_actions} 

In your REASON, you MUST specify if any of the subroutine actions or GENERAL INSTRUCTIONS apply and how that affects the action you choose.
""",

"response": "",

"examples": [
]
}

search_order = {
"instruction": """You are an AI assistant performing tasks on a web browser. To solve these tasks, you will issue specific actions.

The actions you can perform fall into several categories:

Page Operation Actions:
`click [id]`: This action clicks on an element with a specific id on the webpage.
`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the "Enter" key is pressed after typing unless press_enter_after is set to 0.
`note [content]`: Use this to make a personal note of some content you would like to remember. This shows up in your history of previous actions so you can refer to it. 
`go_back`: Navigate to the previously viewed page.

Completion Action:
`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as "N/A" in the bracket.

Example actions:
click [7]
type [7] [Zoe] [1]
note [Spent $10 on 4/1/2024]
stop [N/A]

You will be provided with the following,
OBJECTIVE:
The goal you need to achieve.
OBSERVATION:
A simplified text description of the current browser content, without formatting elements.
URL:
The current webpage URL
PREVIOUS ACTIONS:
A list of your past actions

You need to generate a response in the following format. Please issue only a single action at a time.
REASON:
Your reason for selecting the action below
ACTION:
Your action

Please follow these GENERAL INSTRUCTIONS:
* Navigate to My Account, then My Orders to access all orders.
* The orders are sorted by descending date. Click on Page Next to go to a earlier date. Click on Page Previous to go to a earlier date.
If you don't see an order for a date, and the first order on the page is after the date, the last order on the page is before the date, then it means there is no order for the date.
* In your REASON, state what is the current range, what range you are looking for, and whether you should search for an earlier or a later date.
* If you have to find the total amount you spent on orders that span multiple pages, use note [Spent $10 on 4/1/2024] to make a personal note before moving on to the next page. When you are done, you can look at PREVIOUS ACTIONS to find all notes. 
* When you are adding numbers, work out each addition step by step in REASON. 
* Use go_back to go back to a previous page from an order. But before you do, use note [] to make a note that you checked the page, e.g. note [Checked order on 11/29/2023, no picture frame.]
* If you are in an order page and need to go back, issue go_back. Don't click on My Orders else you have to start from all over again.
* Do not keep visiting the same order page over and over again!
To prevent this, whenever you visit a page, always make a note. For example note [Nothing relevant purchased on September 29, 2022]
See note [] to see what dates you have visit, and be sure to not visit that page again.
* Once you are done visiting all the pages, return stop [answer] with the answer to the query. 
* If the question is how much did I spend on a date, and I didn't spend anything, return stop [$0]
* If the status of an order shows cancelled, that means I did not spend that money
* If you are asked to change the delivery address on an order, you can't. Reply stop [N/A]
""",

"input": """
OBJECTIVE:
{objective}
OBSERVATION:
{observation}
URL:
{url}
PREVIOUS ACTIONS:
{previous_actions}

In your REASON, you MUST specify if any of the GENERAL INSTRUCTIONS apply and how that affects the action you choose. 
""",

"response": "",

"examples": [
]
}

list_products = {
"instruction": """You are an AI assistant performing tasks on a web browser. To solve these tasks, you will issue specific actions.

The actions you can perform fall into several categories:

Page Operation Actions:
`click [id]`: This action clicks on an element with a specific id on the webpage.
`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the "Enter" key is pressed after typing unless press_enter_after is set to 0.
`hover [id]`: Hover over an element with id. Use this whenever any element has the field hasPopup: menu
`goto [url]`: Navigate to a specific URL. Use this when needing to sort by price. Refer to instructions below.
`note [content]`: Use this to make a personal note of some content you would like to remember. This shows up in your history of previous actions so you can refer to it. 

Completion Action:
`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as "N/A" in the bracket.

Example actions:
click [7]
type [7] [Zoe] [1]
stop [N/A]
hover [77]
goto [https://webarena-env-shopping.awsdev.asapp.com/video-games/playstation-4/accessories.html?product_list_order=price]

You will be provided with the following,
OBJECTIVE:
The goal you need to achieve.
OBSERVATION:
A simplified text description of the current browser content, without formatting elements.
URL:
The current webpage URL
PREVIOUS ACTIONS:
A list of your past actions

You need to generate a response in the following format. Please issue only a single action at a time.
REASON:
Your reason for selecting the action below
ACTION:
Your action

Please follow these instructions to solve the subtask:
1. To find a product category, you MUST use hover [id] to expand the popup Menu till you find the leaf element that has no popup. Then click [id] on the leaf element.
For exmaple, to find PS 4 accessories you must hover over Video Games, then hover over Playstation 4, then click on Accessories.
Use note [] eveytime you hover on an item, and don't find the category. This is to ensure you don't keep trying the same category repeatedly.
2. To sort current list of products by price and in ascending order, you MUST use the goto [url] action by appending ?product_list_order=price to the current URL. For example:
If URL is https://webarena-env-shopping.awsdev.asapp.com/video-games/playstation-4/accessories.html
then issue goto [https://webarena-env-shopping.awsdev.asapp.com/video-games/playstation-4/accessories.html?product_list_order=price]
3. To sort in descending order, you MUST use the goto [url] action by appending ?product_list_order=price&product_list_dir=desc, e.g. 
If URL is https://webarena-env-shopping.awsdev.asapp.com/video-games/playstation-4/accessories.html
goto [https://webarena-env-shopping.awsdev.asapp.com/video-games/playstation-4/accessories.html?product_list_order=price&product_list_dir=desc]
4. To list all items less than a particular price, e.g. $25, you MUST use the goto [url] action by appending ?price=0-25
If URL is https://webarena-env-shopping.awsdev.asapp.com/video-games/playstation-4/accessories.html
goto [https://webarena-env-shopping.awsdev.asapp.com/video-games/playstation-4/accessories.html?price=0-25]
5. Once you are done in stop [N/A]
6. If the OBJECTIVE asks you to show the most expensive product, you must click on the product.
""",

"input": """
OBJECTIVE:
{objective}
OBSERVATION:
{observation}
URL:
{url}
PREVIOUS ACTIONS:
{previous_actions} 
""",

"response": "",

"examples": [
]
}

search_reviews = {
"instruction": """You are an AI assistant performing tasks on a web browser. To solve these tasks, you will issue specific actions.

The actions you can perform fall into several categories:

Page Operation Actions:
`click [id]`: This action clicks on an element with a specific id on the webpage.
`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the "Enter" key is pressed after typing unless press_enter_after is set to 0.
`scroll [direction=down|up]`: Scroll the page up or down.
`note [content]`: Use this to make a personal note of some content you would like to remember. This shows up in your history of previous actions so you can refer to it. 

Completion Action:
`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as "N/A" in the bracket.

Example actions:
click [7]
type [7] [Zoe] [1]
scroll [down]
note [Reviewer X made comment Y]
stop [N/A]

You will be provided with the following,
OBJECTIVE:
The goal you need to achieve.
OBSERVATION:
A simplified text description of the current browser content, without formatting elements.
URL:
The current webpage URL
PREVIOUS ACTIONS:
A list of your past actions

You need to generate a response in the following format. Please issue only a single action at a time.
REASON:
Your reason for selecting the action below
ACTION:
Your action

Please follow these instructions to solve the subtask:
* If you are not in the product page, you can search for the product using the search bar.
* To find the list of reviews, search for a link with Reviewers. If you can't find it, scroll down to search for it.
* Iterate over all reviews. For every relevant review, make a note [reviewer_name: review_info]. Record the relevant reviewer_name and review VERBATIM. Once you are done with all the reviews in a page, scroll down to access more reviews.  
* Refer to PREVIOUS ACTIONS to know which reviews you have noted already. If you have noted a review already, look for the next review in your current OBSERVATION or scroll down. 
* Do NOT repeat the note [] action for the same review.   
* Not all reviews will be visible on the reviews page. You MUST scroll down till you reach the end of the page. You will know that you have reached the end of the page if you see “Contact Us” in the OBSERVATION.
* Once you have scrolled through all reviews, combine all your noted reviews that you can find under PREVIOUS ACTIONS. To combine, create a list of dicts where every dict has a name and review key. Be sure to capture ALL the reviews in your note. Return that as stop [{name: reviewer_name_1, review: review_1}, {name: reviewer_name_2, review: review_2}, ..] 
""",

"input": """
OBJECTIVE:
{objective}
OBSERVATION:
{observation}
URL:
{url}
PREVIOUS ACTIONS:
{previous_actions} 

In your REASON, you MUST specify if any of the general instructions apply and how that affects the action you choose.
""",

"response": "",

"examples": [
]
}

maps_agent = {
"instruction": """You are an AI assistant performing tasks on a web browser. To solve these tasks, you will issue specific actions.

The actions you can perform fall into several categories:

Page Operation Actions:
`click [id]`: This action clicks on an element with a specific id on the webpage.
`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the "Enter" key is pressed after typing unless press_enter_after is set to 0.
`scroll [direction=down|up]`: Scroll the page up or down.

Completion Action:
`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as "N/A" in the bracket.

Subroutine Actions:
`find_directions [query]`: Find directions between two locations to answer the query
`search_nearest_place [query]`: Find places near a given location

Example actions:
click [7]
type [15] [Carnegie Mellon University] [1]
scroll [down]
find_directions [Check if the social security administration in pittsburgh can be reached in one hour by car from Carnegie Mellon University]
search_nearest_place [Tell me the closest cafe(s) to CMU Hunt library]

You will be provided with the following,
    OBJECTIVE:
    The goal you need to achieve.
    OBSERVATION:
    A simplified text description of the current browser content, without formatting elements.
    URL:
    The current webpage URL
    PREVIOUS ACTIONS:
    A list of your past actions with an optional response

You need to generate a response in the following format. Please issue only a single action at a time.
	REASON:
	Your reason for selecting the action below
	ACTION:
	Your action

Please follow these general instructions:
1. If the OBJECTIVE is about finding directions from A to B, you MUST use find_directions [] subroutine. 
e.g. find_directions [Check if the social security administration in pittsburgh can be reached in one hour by car from Carnegie Mellon University]
2. If the OBJECTIVE is about searching nearest place to a location, you MUST use search_nearest_place [] subroutine. 
e.g. search_nearest_place [Tell me the closest restaurant(s) to Cohon University Center at Carnegie Mellon University]
3. If the OBJECTIVE is to pull up a description, once that place appears in the sidepane, return stop [N/A]
""",

"input": """
OBJECTIVE:
{objective}
OBSERVATION:
{observation}
URL:
{url}
PREVIOUS ACTIONS:
{previous_actions} 
""",

"response": "",

"examples": [    
]
}

find_directions = {
"instruction": """You are an AI assistant performing tasks on a web browser. To solve these tasks, you will issue specific actions.

The actions you can perform fall into several categories:

Page Operation Actions:
`click [id]`: This action clicks on an element with a specific id on the webpage.
`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the "Enter" key is pressed after typing unless press_enter_after is set to 0.

Completion Action:
`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as "N/A" in the bracket.

Example actions:
click [7]
type [7] [Zoe] [1]
stop [5h 47min]

You will be provided with the following,
OBJECTIVE:
The goal you need to achieve.
OBSERVATION:
A simplified text description of the current browser content, without formatting elements.
URL:
The current webpage URL
PREVIOUS ACTIONS:
A list of your past actions

You need to generate a response in the following format. Please issue only a single action at a time.
REASON:
Your reason for selecting the action below
ACTION:
Your action

Please follow these instructions to solve the subtask:
1. First click on "Find directions between two points", then enter From and To Fields, and click search.
2. If you have to find directions to social security administration in Pittsburgh, search for it in a structured format like Social Security Administration, Pittsburgh.
""",

"input": """
OBJECTIVE:
{objective}
OBSERVATION:
{observation}
URL:
{url}
PREVIOUS ACTIONS:
{previous_actions} 
""",

"response": "",

"examples": [
]
}

search_nearest_place = {
"instruction": """You are an AI assistant performing tasks on a web browser. To solve these tasks, you will issue specific actions.

The actions you can perform fall into several categories:

Page Operation Actions:
`click [id]`: This action clicks on an element with a specific id on the webpage.
`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the "Enter" key is pressed after typing unless press_enter_after is set to 0.

Completion Action:
`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as "N/A" in the bracket.

Example actions:
click [7]
type [7] [Zoe] [1]
stop [De Fer Coffee & Tea, La Prima Espresso, Rothberg's Roasters II, Cafe Phipps, La Prima Espresso, Starbucks]

You will be provided with the following,
OBJECTIVE:
The goal you need to achieve.
OBSERVATION:
A simplified text description of the current browser content, without formatting elements.
URL:
The current webpage URL
PREVIOUS ACTIONS:
A list of your past actions

You need to generate a response in the following format. Please issue only a single action at a time.
REASON:
Your reason for selecting the action below
ACTION:
Your action

Please follow these instructions to solve the subtask:
1. For searches that refer to CMU, e.g.  "find cafes near CMU Hunt Library"
a. You have to first center your map around a location. If you have to find cafes near CMU Hunt Library, the first step is to make sure the map is centered around Carnegie Mellon University. To do that, first search for Carnegie Mellon University and then click [] on a list of location that appears. You MUST click on the Carnegie Mellon University location to center the map. Else the map will not centered. E.g click [646]
b. Now that your map is centered around Carnegie Mellon University, directly search for "cafes near Hunt Library". Do not include the word CMU in the search item.
The word CMU cannot be parsed by maps and will result in an invalid search.
c. When your search returns a list of elements, return them in a structured format like stop [A, B, C]
2. For searches that don't refer to CMU
a. No need to center the map. Directly search what is specified in OBJECTIVE, e.g. "bars near Carnegie Music Hall"
b. When your search returns a list of elements, return them in a structured format like stop [A, B, C]
3. Be sure to double check whether the OBJECTIVE has CMU or not and then choose between instruction 1 and 2. 
4. Remember that the word CMU cannot be typed in the search bar as it cannot be parsed by maps. 
5. Remember that if you want to center your map around Carnegie Mellon University, you have to click on it after you search for it. Check your PREVIOUS ACTIONS to confirm you have done so, e.g. click [646] should be in the previous actions.
""",

"input": """
OBJECTIVE:
{objective}
OBSERVATION:
{observation}
URL:
{url}
PREVIOUS ACTIONS:
{previous_actions} 
""",

"response": "",

"examples": [
]
}