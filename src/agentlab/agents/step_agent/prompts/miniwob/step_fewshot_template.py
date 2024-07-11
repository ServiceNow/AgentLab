miniwob_agent = {
"instruction": """You are an AI assistant performing tasks on a web browser. To solve these tasks, you will issue specific actions.

The actions you can perform fall into several categories:

Low Level Actions:
click [id]: Clicks an element corresponding to the provided id.
type [id] [content]: Types the provided content into the element corresponding to the provided id.
stop [answer]: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. Otherwise, leave it empty.

Subroutine Actions:
fill_text [id] [text]: Fill a text box with the given text.
choose_date [id] [date]: Select a date value from a datepicker.
find_and_click_search_link [link_number]: When dealing with a search engine, find and click a link the link_number from search results.
click_checkboxes [objective]: Select checkbox(es) specified by objective
process_email [objective]: Find / Reply / Process any email task specified by objective

Please issue only one action at a time.

Example actions:
click [7]
fill_text [flight-from] [LEB]
choose_date [datepicker] [12/26/2016]
find_and_click_search_link [7]
click_checkboxes [Select words similar to mild, delicious and click Submit]
process_email [Give Bobine the email you received from Cora]

You will be provided with the following,
    OBJECTIVE:
    The goal you need to achieve.
    OBSERVATION:
    A simplified text description of the current browser content, without formatting elements.
    URL:
    The current webpage URL
    PREVIOUS ACTIONS:
    A list of your past actions

You need to generate response containing,
    REASON:
    A rationale for selecting the action below
    ACTION:
    A single action
 
Please follow these instructions:

1. When interacting with a datepicker, only select action choose_date and not other actions.
2. Only when doing tasks involving a search engine, select action find_and_click_search_link. Do not use otherwise.
3. When asked to select nothing, make sure you are not clicking on any checkboxes.
4. If the task requires you to select checkboxes, select action click_checkboxes [objective] where [objective] is the original objective. 
E.g. click_checkboxes [Select words similar to mild, delicious and click Submit]
5. If the task requires you to find emails or send emails or forward emails or reply to emails, select action process_email [objective] where  objective is the original objective. Directly issue process_email. 

OBJECTIVE:
Give Bobine the email you received from Cora
REASON:
I will first check the instruction to see if the OBJECTIVE falls in that list. Yes it does: If the task requires you to find emails or send emails or forward emails or reply to emails, select action process_email [objective]
ACTION:
process_email [Give Bobine the email you received from Cora]

6. First generate REASON: then generate ACTION:
7. Begin all REASON with 'I will first check the instruction to see if the OBJECTIVE falls in that list' and then check the instructions above.
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
Book the shortest one-way flight from: LEB to: RDG on 12/26/2016.
OBSERVATION:
<body id=1 val= />
<div id=2 val=wrap />
<div id=3 val=area />
<div id=4 val=menu />
<h2 id=5 val=Book Your One-Way Flight />
<div id=6 val= />
<input_text id=7 val=flight-from />
<div id=8 val= />
<input_text id=9 val=flight-to />
<div id=10 val= />
<div id=11 val=Departure Date />
<div id=12 val= />
<input_text id=13 val=datepicker />
<button id=15 val=Search />
URL:
PREVIOUS ACTIONS:
fill_text [flight-from] [LEB]
fill_text [flight-to] [RDG]
choose_date [datepicker] [12/26/2016]
""",
"response": """
REASON:
Looking at my previous actions, I have already filled in flight from and flight to and the date. I have to next click on search.
ACTION:
click [15]
"""},
{
"input": """ 
OBJECTIVE:
Copy the text in the textarea below, paste it into the textbox and press Submit.
OBSERVATION:
<body id=1 val= />
<div id=2 val=wrap />
<div id=3 val=area />
<div id=4 val=container />
<input_text id=5 val=/>
<textarea id=6 val=Commodo adipiscing eu erat enim ipsum sodales eget id />
<button id=7 val=Submit />
URL:
PREVIOUS ACTIONS:

""",
"response": """
REASON:
I have to first copy the text from text area id=6 to input_text id=5
ACTION:
fill_text [5] [Commodo adipiscing eu erat enim ipsum sodales eget id]
"""
},
{
"input": """ 
OBJECTIVE:
Find the 2nd word in the paragraph, type that into the textbox and press "Submit".
OBSERVATION:
<body id=1 val= />
<div id=2 val=wrap />
<div id=3 val=area />
<p id=4 val=Dis urna proin placerat neque, lectus turpis. />
<input_text id=5 val=answer-input />
<button id=6 val=Submit />
URL:
PREVIOUS ACTIONS:

""",
"response": """
REASON:
I have to find the 2nd word in the paragraph and type it into the textbox.
The paragraph is in id 4, "Dis urna proin placerat neque, lectus turpis."
The 2nd word is "urna".
I have to type that in id 5.
ACTION:
fill_text [5] [urna]
"""
},
]
}

fill_text = {
"instruction": """You are an AI assistant performing tasks on a web browser. To solve these tasks, you will issue specific actions.

You can only interact with web elements like links, inputs, buttons in the browser content. You can issue any one of the actions below:
click [id]: Clicks an element corresponding to the provided id.
type [id] [content]: Types the provided content into the element corresponding to the provided id.
stop [answer]: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. Otherwise, leave it empty.

Examples of actions are click [7], type [11] [New York]. Please issue only one action at a time.

You will be provided with the following,
    OBJECTIVE:
    The goal you need to achieve.
    OBSERVATION:
    A simplified text description of the current browser content, without formatting elements.
    URL:
    The current webpage URL
    PREVIOUS ACTIONS:
    A list of your past actions

You need to generate response containing,
    REASON:
    A rationale for selecting the action below
    ACTION:
    A single action
    
Please follow these instructions
1. First type in a text box.
2. If there is a dropdown text, click on the corresponding id.
3. Issue stop [] when done.
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
fill_text [flight-from] [LEB]
OBSERVATION:
<body id=1 val= />
<div id=2 val=wrap />
<div id=3 val=area />
<div id=4 val=menu />
<h2 id=5 val=Book Your One-Way Flight />
<div id=6 val= />
<input_text id=7 val=flight-from />
<div id=8 val= />
<input_text id=9 val=flight-to />
<div id=10 val= />
URL:
PREVIOUS ACTIONS:

""",
"response": """
REASON:
I have no previous actions.
I have to first type "LEB" in the field flight-from corresponding to id 7
ACTION:
type [7] [LEB]
"""},
{
"input": """
OBJECTIVE:
fill_text [flight-from] [LEB]
OBSERVATION:
<body id=1 val= />
<div id=2 val=wrap />
<div id=3 val=area />
<div id=4 val=menu />
<h2 id=5 val=Book Your One-Way Flight />
<div id=6 val= />
<input_text id=7 val=flight-from />
<div id=8 val= />
<div id=14 val= />
<button id=15 val=Search />
<ul id=18 val=ui-id-1 />
<li id=19 val= />
<div id=20 val=Hanover, NH (HNV) />
<li id=21 val= />
<div id=22 val=Lebanon, NH (LEB) />
<li id=23 val= />
<div id=24 val=White River, VT (WHR) />
<div id=16 val= />
<div id=25 val=3 results are available, use up and down arrow keys to navigate. />
<div id=17 val= />
URL:
PREVIOUS ACTIONS:
type [7] [LEB]
""",
"response": """
REASON:
I have already typed in "LEB" in id 7
There is a corresponding dropdown text in "Lebanon, NH (LEB)" in id 22
I have to click on id 22 Lebanon, NH (LEB)
ACTION:
click [22]
"""}
]
}

choose_date = {
"instruction": """You are an AI assistant performing tasks on a web browser. To solve these tasks, you will issue specific actions.

You can only interact with web elements like links, inputs, buttons in the browser content. You can issue any one of the actions below:
click [id]: Clicks an element corresponding to the provided id.
type [id] [content]: Types the provided content into the element corresponding to the provided id.
stop [answer]: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. Otherwise, leave it empty.

Examples of actions are click [7], type [11] [New York]. Please issue only one action at a time.

You will be provided with the following,
    OBJECTIVE:
    The goal you need to achieve.
    OBSERVATION:
    A simplified text description of the current browser content, without formatting elements.
    URL:
    The current webpage URL
    PREVIOUS ACTIONS:
    A list of your past actions

You need to generate response containing,
    REASON:
    A rationale for selecting the action below
    ACTION:
    A single action

Please follow these general instructions
1. First click on the datepicker
2. Check if you are in the right month, else use Prev to Navigate to the right month 
3. Once you are at the right month, click on the right date
4. Once you have clicked the date, issue stop [].
5. Alternatively, if there is no datepicker in the browser context, directly type the date. 
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
choose_date [datepicker] [11/03/2016]
OBSERVATION:
<div id=8 val= />
<a id=9 val= />
<span id=10 val=Prev />
<a id=11 val= />
<div id=13 val= />
<span id=14 val=December />
<span id=15 val=2016 />
<a id=40 val=12/1/2016 />
<a id=42 val=12/2/2016 />
<a id=44 val=12/3/2016 />
<a id=47 val=12/4/2016 />
<a id=49 val=12/5/2016 />
<a id=51 val=12/6/2016 />
<a id=53 val=12/7/2016 />
<a id=55 val=12/8/2016 />
<a id=57 val=12/9/2016 />
<a id=59 val=12/10/2016 />
<a id=62 val=12/11/2016 />
<a id=64 val=12/12/2016 />
<a id=66 val=12/13/2016 />
<a id=68 val=12/14/2016 />
URL:
PREVIOUS ACTIONS:
click [5] 
""",
"response": """
REASON:
I have already clicked on datepicker.
Looking at the current browser content val, I am currently in Decemeber (12/2016). 
I have to go to November (11/2016). 
Since 11 < 12, I have to click on Prev
ACTION:
click [10]
"""},
{
"input": """
OBJECTIVE:
choose_date [datepicker] [11/03/2016]
OBSERVATION:
<tbody id=33 val= />
<a id=40 val=11/1/2016 />
<a id=42 val=11/2/2016 />
<a id=44 val=11/3/2016 />
<a id=47 val=11/4/2016 />
<a id=49 val=11/5/2016 />
<a id=51 val=11/6/2016 />
<a id=53 val=11/7/2016 />
<a id=55 val=11/8/2016 />
<a id=57 val=11/9/2016 />
<a id=59 val=11/10/2016 />
<a id=62 val=11/11/2016 />
<a id=64 val=11/12/2016 />
<a id=66 val=11/13/2016 />
<a id=68 val=11/14/2016 />
<a id=70 val=11/15/2016 />
URL:
PREVIOUS ACTIONS:
click [5] 
click [10]
""",
"response": """
REASON:
I have already clicked on datepicker.
Looking at the current browser content val, I am currently in November (11/2016). 
I have to go to November (11/2016). 
Since 11 = 11, I am in the correct month.
I have to click on the id corresponding to 11/3/2016
ACTION:
click [44]
"""},
{
"input": """
OBJECTIVE:
choose_date [5] [05/20/2010]
OBSERVATION:
<body id=1 val= />
<div id=2 val=wrap />
<div id=3 val=area />
<div id=4 val=form />
<input_date id=5 val=tt />
<button id=6 val=Submit />
URL:
PREVIOUS ACTIONS:
""",
"response": """
REASON:
I see no datepicker. I will directly type the date.
ACTION:
type [5] [05/20/2010]
"""},
]
}

find_and_click_search_link = {
"instruction": """You are an AI assistant performing tasks on a web browser. To solve these tasks, you will issue specific actions.

You can only interact with web elements like links, inputs, buttons in the browser content. You can issue any one of the actions below:
click [id]: Clicks an element corresponding to the provided id.
type [id] [content]: Types the provided content into the element corresponding to the provided id.
stop [answer]: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. Otherwise, leave it empty.

Examples of actions are click [7], type [11] [New York]. Please issue only one action at a time.

You will be provided with the following,
    OBJECTIVE:
    The goal you need to achieve.
    OBSERVATION:
    A simplified text description of the current browser content, without formatting elements.
    URL:
    The current webpage URL
    PREVIOUS ACTIONS:
    A list of your past actions

You need to generate response containing,
    REASON:
    A rationale for selecting the action below
    ACTION:
    A single action

Please follow these general instructions:
1. Your goal is through search links, navigating webpages till you find the link referred to in the OBJECTIVE. 
2. Keep count of the number of links in the webpage
3. Navigate till you find the link referred to in the OBJECTIVE
4. Click on the link
5. Issue stop []
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
find_and_click_search_link [7]
OBSERVATION:
<a id=9 val=Juan />
<div id=10 val=https://www.puruspurus.org />
<div id=11 val=Tincidunt quis velit. />
<div id=12 val= />
<a id=13 val=Donovan />
<div id=14 val=https://www.mi.jp />
<div id=15 val=Purus feugiat. />
<div id=16 val= />
<a id=17 val=Alan />
<div id=18 val=https://fermentum.it />
<div id=19 val=Semper pretium. />
<ul id=20 val=pagination />
<li id=21 val= />
<a id=22 val=1 />
<li id=23 val= />
<a id=24 val=2 />
<li id=25 val= />
<a id=26 val=3 />
<li id=27 val= />
<a id=28 val=> />
URL:
PREVIOUS ACTIONS:
""",
"response": """
REASON:
I have no previous actions.
Since I have no previous actions, I have seen a total of 0 links in the past. 
Looking at the browser content, I see the following links: Juan, Donovan, Alan
The makes the index of the links as Juan (0 + 1 = 1), Donovan (0 + 2 = 2), Alan (0 + 3 = 3)
I am told to find and click the link 7. 
Since 7 > 3, I have to go to the next page.
To go to the next page, I have to click on the id corresponding to > sign, which corresponds to id 28.
I will click on id 28
ACTION:
click [28]
"""},
{
"input": """
OBJECTIVE:
find_and_click_search_link [7]
OBSERVATION:
<a id=30 val=Michel />
<div id=31 val=https://www.netus.ca />
<div id=32 val=Nascetur aliquet. />
<div id=33 val= />
<a id=34 val=Cheree />
<div id=35 val=https://nullalectus.hk />
<div id=36 val=Venenatis, ac. />
<div id=37 val= />
<a id=38 val=Briana />
<div id=39 val=https://turpis.pizza />
<div id=40 val=Scelerisque a duis. />
<ul id=20 val=pagination />
<li id=41 val= />
<a id=42 val=< />
<li id=43 val= />
<a id=44 val=1 />
<li id=45 val= />
<a id=46 val=2 />
<li id=47 val= />
<a id=48 val=3 />
<li id=49 val= />
<a id=50 val=> />
URL:
PREVIOUS ACTIONS:
click [28]
""",
"response": """
REASON:
I have 1 previous action
Since I have 1 previous action, I have seen a total of 3 links in the past. 
Looking at the browser content, I see the following links: Michel, Cheree , Briana
The makes the index of the links as Michel (3 + 1 = 4), Cheree (3 + 2 = 5), Briana (3 + 3 = 6)
I am told to find and click the link 7. 
Since 7 > 6, I have to go to the next page.
To go to the next page, I have to click on the id corresponding to > sign, which corresponds to id 50.
I will click on id 50
ACTION:
click [50]
"""},
{
"input": """
OBJECTIVE:
find_and_click_search_link [7]
OBSERVATION:
<a id=52 val=Renda />
<div id=53 val=https://estinteger.it />
<div id=54 val=Porttitor. Quis. />
<div id=55 val= />
<a id=56 val=Donovan />
<div id=57 val=https://volutpatsit.it />
<div id=58 val=Magnis. Arcu aliquam. />
<div id=59 val= />
<a id=60 val=Livia />
<div id=61 val=https://www.lectussed.eu />
<div id=62 val=Ornare sit vulputate. />
<ul id=20 val=pagination />
<li id=63 val= />
<a id=64 val=< />
<li id=65 val= />
<a id=66 val=1 />
<li id=67 val= />
<a id=68 val=2 />
<li id=69 val= />
<a id=70 val=3 />
URL:
PREVIOUS ACTIONS:
click [28]
click [50]
""",
"response": """
REASON:
I have 2 previous action
Since I have 2 previous action, I have seen a total of 6 links in the past. 
Looking at the browser content, I see the following links: Renda, Donovan, Livia
The makes the index of the links as Renda (6 + 1 = 7), Donovan (6 + 2 = 8), Livia (6 + 3 = 9)
I am told to find and click the link 7. 
Since 7=7, I have found the link I am looking for.
The 7th link corresponds to Renda
I am going to click on Renda which corresponds to id 52
ACTION:
click [52]
"""},
]
}

click_checkboxes = {
"instruction": """You are an AI assistant performing tasks on a web browser. To solve these tasks, you will issue specific actions.

You can only interact with web elements like links, inputs, buttons in the browser content. You can issue any one of the actions below:
click [id]: Clicks an element corresponding to the provided id.
type [id] [content]: Types the provided content into the element corresponding to the provided id.
stop [answer]: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. Otherwise, leave it empty.

Examples of actions are click [7], type [11] [New York]. Please issue only one action at a time.

You will be provided with the following,
    OBJECTIVE:
    The goal you need to achieve.
    OBSERVATION:
    A simplified text description of the current browser content, without formatting elements.
    URL:
    The current webpage URL
    PREVIOUS ACTIONS:
    A list of your past actions

You need to generate response containing,
    REASON:
    A rationale for selecting the action below
    ACTION:
    A single action
    
    
Please follow these general instructions:
1. Your goal is click on all the checkboxes mentioned in the OBJECTIVE.
2. Once you are done issue stop []
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
click_checkboxes [Select words similar to mild, delicious and click Submit]
OBSERVATION:
<body id=1 val= />
<div id=2 val=wrap />
<div id=3 val=area />
<div id=4 val=boxes />
<label id=5 val= />
<input_checkbox id=6 val=ch0 />
<t id=-1 val=stop />
<label id=7 val= />
<input_checkbox id=8 val=ch1 />
<t id=-2 val=archaic />
<label id=9 val= />
<input_checkbox id=10 val=ch2 />
<t id=-3 val=quiet />
<label id=11 val= />
<input_checkbox id=12 val=ch3 />
<t id=-4 val=delectable />
<label id=13 val= />
<input_checkbox id=14 val=ch4 />
<t id=-5 val=fire />
<label id=15 val= />
<input_checkbox id=16 val=ch5 />
<t id=-6 val=sinful />
<button id=17 val=Submit />
URL:
PREVIOUS ACTIONS:
""",
"response": """
REASON:
I have to select words similar to mild, delicious and click Submit.
Looking at the browser content, I see 6 checkboxes with words stop, archaic, quiet, delectable, fire, sinful.
The closest word similar to mild is quiet (id=10).
The closest word similar to delicious is delectable (id=12).
So, I have to select checkboxes with ids 10 and 12
Finally, I have to click Submit (id=17).
I have no past actions, so I must start with the first, i.e. click 10
ACTION:
click [10]
"""},
]
}

process_email = {
"instruction": """You are an AI assistant performing tasks on a web browser. To solve these tasks, you will issue specific actions.

You can only interact with web elements like links, inputs, buttons in the browser content. You can issue any one of the actions below:
click [id]: Clicks an element corresponding to the provided id.
type [id] [content]: Types the provided content into the element corresponding to the provided id.
stop [answer]: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. Otherwise, leave it empty.

Examples of actions are click [7], type [11] [New York]. Please issue only one action at a time.

You will be provided with the following,
    OBJECTIVE:
    The goal you need to achieve.
    OBSERVATION:
    A simplified text description of the current browser content, without formatting elements.
    URL:
    The current webpage URL
    PREVIOUS ACTIONS:
    A list of your past actions

You need to generate response containing,
    REASON:
    A rationale for selecting the action below
    ACTION:
    A single action
 
Please follow these general instructions:
1. Your goal is to navigate an email server to solve the OBJECTIVE
2. Once you have solved it, issue stop []
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
process_email [Give Bobine the email you received from Cora]
OBSERVATION:
<div id=9 val= />
<div id=10 val=Audrey />
<div id=11 val=Ridiculus eget... />
<div id=12 val=Imperdiet. Curs.. />
<div id=13 val= />
<div id=18 val= />
<div id=19 val=Cora />
<div id=20 val=In id. />
<div id=21 val=Lacus. At sit. .. />
<div id=22 val= />
<div id=27 val= />
<div id=28 val=Bobine />
<div id=29 val=Cras. Convallis.. />
<div id=30 val=Purus feugiat. .. />
<div id=31 val= />
URL:
PREVIOUS ACTIONS:
""",
"response": """
REASON:
I have to forward Bobine the email I received from Cora.
Looking at the browser content, I can see that Cora's email is in id 21, "Lacus. At sit. .."
I have to click on that email first so I can forward it to Bobine.
ACTION:
click [21]
"""},
{
"input": """
OBJECTIVE:
process_email [Give Bobine the email you received from Cora]
OBSERVATION:
<div id=35 val=email />
<div id=36 val=email-bar />
<span id=37 val=close-email />
<div id=41 val= />
<div id=42 val=In id. />
<span id=43 val= />
<div id=44 val=Cora />
<div id=45 val=to me />
<span id=46 val= />
<div id=47 val=Lacus. At sit. Volutpat tellus. Maecenas commodo, purus pellentesque tellus duis leo pulvinar varius. />
<div id=48 val= />
<div id=51 val=Reply />
<div id=53 val= />
<div id=54 val=Forward />
URL:
PREVIOUS ACTIONS:
click [21]
""",
"response": """
REASON:
I have to forward Bobine the email I received from Cora.
Based on my past actions, I have already clicked on the email.
I now have to click on the Forward button.
ACTION:
click [54]
"""},
{
"input": """
OBJECTIVE:
process_email [Give Bobine the email you received from Cora]
OBSERVATION:
<body id=1 val= />
<div id=2 val=wrap />
<div id=3 val=area />
<div id=55 val=forward />
<div id=56 val=forward-bar />
<span id=57 val=close-forward />
<span id=58 val=send-forward />
<div id=59 val= />
<label id=61 val=to: />
<input_text id=62 val= />
<div id=63 val= />
<label id=64 val=subject: />
<t id=-1 val=In id. />
<div id=65 val= />
<textarea id=66 val=Lacus. At sit. Volutpat tellus. Maecenas commodo, purus pellentesque tellus duis leo pulvinar varius. />
URL:
PREVIOUS ACTIONS:
click [21]
click [54]
""",
"response": """
REASON:
I have to forward Bobine the email I received from Cora.
Based on my past actions, I have already clicked on the email from Cora, and clicked the Forward button.
Lookinat the the browser content, the to: field is empty.
I have to enter Bobine in the to: field. 
After that I can send the forward.
ACTION:
type [62] [Bobine]
"""}
]
}