Models & Pricing
The prices listed below are in unites of per 1M tokens. A token, the smallest unit of text that the model recognizes, can be a word, a number, or even a punctuation mark. We will bill based on the total number of input and output tokens by the model.

Pricing Details
USD
CNY
MODEL(1)	CONTEXT LENGTH	MAX COT TOKENS(2)	MAX OUTPUT TOKENS(3)	1M TOKENS
INPUT PRICE
(CACHE HIT) (4)	1M TOKENS
INPUT PRICE
(CACHE MISS)	1M TOKENS
OUTPUT PRICE
deepseek-chat	64K	-	8K	$0.07(5)
$0.014	$0.27(5)
$0.14	$1.10(5)
$0.28
deepseek-reasoner	64K	32K	8K	$0.14	$0.55	$2.19 (6)
(1) The deepseek-chat model has been upgraded to DeepSeek-V3. deepseek-reasoner points to the new model DeepSeek-R1.
(2) CoT (Chain of Thought) is the reasoning content deepseek-reasoner gives before output the final answer. For details, please refer to Reasoning Model。
(3) If max_tokens is not specified, the default maximum output length is 4K. Please adjust max_tokens to support longer outputs.
(4) Please check DeepSeek Context Caching for the details of Context Caching.
(5) The form shows the the original price and the discounted price. From now until 2025-02-08 16:00 (UTC), all users can enjoy the discounted prices of DeepSeek API. After that, it will recover to full price. DeepSeek-R1 is not included in the discount.
(6) The output token count of deepseek-reasoner includes all tokens from CoT and the final answer, and they are priced equally.
Deduction Rules
The expense = number of tokens × price. The corresponding fees will be directly deducted from your topped-up balance or granted balance, with a preference for using the granted balance first when both balances are available.

Product prices may vary and DeepSeek reserves the right to adjust them. We recommend topping up based on your actual usage and regularly checking this page for the most recent pricing information.



Your First API Call
The DeepSeek API uses an API format compatible with OpenAI. By modifying the configuration, you can use the OpenAI SDK or softwares compatible with the OpenAI API to access the DeepSeek API.

PARAM	VALUE
base_url *       	https://api.deepseek.com
api_key	apply for an API key
* To be compatible with OpenAI, you can also use https://api.deepseek.com/v1 as the base_url. But note that the v1 here has NO relationship with the model's version.

* The deepseek-chat model has been upgraded to DeepSeek-V3. The API remains unchanged. You can invoke DeepSeek-V3 by specifying model='deepseek-chat'.

* deepseek-reasoner is the latest reasoning model, DeepSeek-R1, released by DeepSeek. You can invoke DeepSeek-R1 by specifying model='deepseek-reasoner'.

Invoke The Chat API
Once you have obtained an API key, you can access the DeepSeek API using the following example scripts. This is a non-stream example, you can set the stream parameter to true to get stream response.

curl
python
nodejs
# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI

client = OpenAI(api_key="<DeepSeek API Key>", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)


The Temperature Parameter
The default value of temperature is 1.0.

We recommend users to set the temperature according to their use case listed in below.
USE CASE	TEMPERATURE
Coding / Math   	0.0
Data Cleaning / Data Analysis	1.0
General Conversation	1.3
Translation	1.3
Creative Writing / Poetry	1.5


Token & Token Usage
Tokens are the basic units used by models to represent natural language text, and also the units we use for billing. They can be intuitively understood as 'characters' or 'words'. Typically, a Chinese word, an English word, a number, or a symbol is counted as a token.

Generally, the conversion ratio between tokens in the model and the number of characters is approximately as following:

1 English character ≈ 0.3 token.
1 Chinese character ≈ 0.6 token.
However, due to the different tokenization methods used by different models, the conversion ratios can vary. The actual number of tokens processed each time is based on the model's return, which you can view from the usage results.

Calculate token usage offline
You can run the demo tokenizer code in the following zip package to calculate the token usage for your intput/output.

deepseek_v3_tokenizer.zip


Rate Limit
DeepSeek API does NOT constrain user's rate limit. We will try out best to serve every request.

However, please note that when our servers are under high traffic pressure, your requests may take some time to receive a response from the server. During this period, your HTTP request will remain connected, and you may continuously receive contents in the following formats:

Non-streaming requests: Continuously return empty lines
Streaming requests: Continuously return SSE keep-alive comments (: keep-alive)
These contents do not affect the parsing of the JSON body by the OpenAI SDK. If you are parsing the HTTP responses yourself, please ensure to handle these empty lines or comments appropriately.

If the request is still not completed after 30 minutes, the server will close the connection.

Error Codes
When calling DeepSeek API, you may encounter errors. Here list the causes and solutions.

                    CODE                    	DESCRIPTION
400 - Invalid Format	Cause: Invalid request body format.
Solution: Please modify your request body according to the hints in the error message. For more API format details, please refer to DeepSeek API Docs.
401 - Authentication Fails	Cause: Authentication fails due to the wrong API key.
Solution: Please check your API key. If you don't have one, please create an API key first.
402 - Insufficient Balance	Cause: You have run out of balance.
Solution: Please check your account's balance, and go to the Top up page to add funds.
422 - Invalid Input_schema	Cause: Your request contains invalid input_schema.
Solution: Please modify your request input_schema according to the hints in the error message. For more API format details, please refer to DeepSeek API Docs.
429 - Rate Limit Reached	Cause: You are sending requests too quickly.
Solution: Please pace your requests reasonably. We also advise users to temporarily switch to the APIs of alternative LLM service providers, like OpenAI.
500 - Server Error	Cause: Our server encounters an issue.
Solution: Please retry your request after a brief wait and contact us if the issue persists.
503 - Server Overloaded	Cause: The server is overloaded due to high traffic.
Solution: Please retry your request after a brief wait.


DeepSeek-R1 Release
⚡ Performance on par with OpenAI-o1

📖 Fully open-source model & technical report

🏆 MIT licensed: Distill & commercialize freely!

🌐 Website & API are live now! Try DeepThink at chat.deepseek.com today!



🔥 Bonus: Open-Source Distilled Models!

🔬 Distilled from DeepSeek-R1, 6 small models fully open-sourced

📏 32B & 70B models on par with OpenAI-o1-mini

🤝 Empowering the open-source community

🌍 Pushing the boundaries of open AI!


📜 License Update!

🔄 DeepSeek-R1 is now MIT licensed for clear open access

🔓 Open for the community to leverage model weights & outputs

🛠️ API outputs can now be used for fine-tuning & distillation

🛠️ DeepSeek-R1: Technical Highlights

📈 Large-scale RL in post-training

🏆 Significant performance boost with minimal labeled data

🔢 Math, code, and reasoning tasks on par with OpenAI-o1

📄 More details: https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf


🌐 API Access & Pricing

⚙️ Use DeepSeek-R1 by setting model=deepseek-reasoner

💰 $0.14 / million input tokens (cache hit)

💰 $0.55 / million input tokens (cache miss)

💰 $2.19 / million output tokens

📖 API guide: https://api-docs.deepseek.com/guides/reasoning_model





API ReferenceChatCreate Chat Completion
Create Chat Completion
POST
https://api.deepseek.com/chat/completions
Creates a model response for the given chat conversation.

Request
application/json
Body

required

messages

object[]

required

model
string
required
Possible values: [deepseek-chat, deepseek-reasoner]

ID of the model to use. You can use deepseek-chat.

frequency_penalty
number
nullable
Possible values: >= -2 and <= 2

Default value: 0

Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.

max_tokens
integer
nullable
Possible values: > 1

Integer between 1 and 8192. The maximum number of tokens that can be generated in the chat completion.

The total length of input tokens and generated tokens is limited by the model's context length.

If max_tokens is not specified, the default value 4096 is used.

presence_penalty
number
nullable
Possible values: >= -2 and <= 2

Default value: 0

Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.

response_format

object

nullable

stop

object

nullable

stream
boolean
nullable
If set, partial message deltas will be sent. Tokens will be sent as data-only server-sent events (SSE) as they become available, with the stream terminated by a data: [DONE] message.

stream_options

object

nullable

temperature
number
nullable
Possible values: <= 2

Default value: 1

What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.

We generally recommend altering this or top_p but not both.

top_p
number
nullable
Possible values: <= 1

Default value: 1

An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.

We generally recommend altering this or temperature but not both.

tools

object[]

nullable

tool_choice

object

nullable

logprobs
boolean
nullable
Whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message.

top_logprobs
integer
nullable
Possible values: <= 20

An integer between 0 and 20 specifying the number of most likely tokens to return at each token position, each with an associated log probability. logprobs must be set to true if this parameter is used.

Responses
200 (No streaming)
200 (Streaming)
OK, returns a chat completion object

application/json
Schema
Example (from schema)
Example
Schema

id
string
required
A unique identifier for the chat completion.

choices

object[]

required

created
integer
required
The Unix timestamp (in seconds) of when the chat completion was created.

model
string
required
The model used for the chat completion.

system_fingerprint
string
required
This fingerprint represents the backend configuration that the model runs with.

object
string
required
Possible values: [chat.completion]

The object type, which is always chat.completion.

usage

object

curl
python
go
nodejs
ruby
csharp
php
java
powershell
OpenAI SDK
from openai import OpenAI

# for backward compatibility, you can still use `https://api.deepseek.com/v1` as `base_url`.
client = OpenAI(api_key="<your API key>", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
  ],
    max_tokens=1024,
    temperature=0.7,
    stream=False
)

print(response.choices[0].message.content)



REQUESTS
HTTP.CLIENT
import requests
import json

url = "https://api.deepseek.com/chat/completions"

payload = json.dumps({
  "messages": [
    {
      "content": "You are a helpful assistant",
      "role": "system"
    },
    {
      "content": "Hi",
      "role": "user"
    }
  ],
  "model": "deepseek-chat",
  "frequency_penalty": 0,
  "max_tokens": 2048,
  "presence_penalty": 0,
  "response_format": {
    "type": "text"
  },
  "stop": None,
  "stream": False,
  "stream_options": None,
  "temperature": 1,
  "top_p": 1,
  "tools": None,
  "tool_choice": "none",
  "logprobs": False,
  "top_logprobs": None
})
headers = {
  'Content-Type': 'application/json',
  'Accept': 'application/json',
  'Authorization': 'Bearer <TOKEN>'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)


Request
Collapse all
Base URL
https://api.deepseek.com
Auth
Bearer Token
Bearer Token
Body
 required
{
  "messages": [
    {
      "content": "You are a helpful assistant",
      "role": "system"
    },
    {
      "content": "Hi",
      "role": "user"
    }
  ],
  "model": "deepseek-chat",
  "frequency_penalty": 0,
  "max_tokens": 2048,
  "presence_penalty": 0,
  "response_format": {
    "type": "text"
  },
  "stop": null,
  "stream": false,
  "stream_options": null,
  "temperature": 1,
  "top_p": 1,
  "tools": null,
  "tool_choice": "none",
  "logprobs": false,
  "top_logprobs": null
}
Send API Request
Response
Clear
Click the Send API Request button above and


API ReferenceCompletionsCreate FIM Completion (Beta)
Create FIM Completion (Beta)
POST
https://api.deepseek.com/beta/completions
The FIM (Fill-In-the-Middle) Completion API. User must set base_url="https://api.deepseek.com/beta" to use this feature.

Request
application/json
Body

required

model
string
required
Possible values: [deepseek-chat]

ID of the model to use.

prompt
string
required
Default value: Once upon a time,

The prompt to generate completions for.

echo
boolean
nullable
Echo back the prompt in addition to the completion

frequency_penalty
number
nullable
Possible values: >= -2 and <= 2

Default value: 0

Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.

logprobs
integer
nullable
Possible values: <= 20

Include the log probabilities on the logprobs most likely output tokens, as well the chosen tokens. For example, if logprobs is 20, the API will return a list of the 20 most likely tokens. The API will always return the logprob of the sampled token, so there may be up to logprobs+1 elements in the response.

The maximum value for logprobs is 20.

max_tokens
integer
nullable
The maximum number of tokens that can be generated in the completion.

presence_penalty
number
nullable
Possible values: >= -2 and <= 2

Default value: 0

Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.

stop

object

nullable

stream
boolean
nullable
Whether to stream back partial progress. If set, tokens will be sent as data-only server-sent events as they become available, with the stream terminated by a data: [DONE] message. Example Python code.

stream_options

object

nullable

suffix
string
nullable
The suffix that comes after a completion of inserted text.

temperature
number
nullable
Possible values: <= 2

Default value: 1

What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.

We generally recommend altering this or top_p but not both.

top_p
number
nullable
Possible values: <= 1

Default value: 1

An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.

We generally recommend altering this or temperature but not both.

Responses
200
OK

application/json
Schema
Example (from schema)
Schema

id
string
required
A unique identifier for the completion.

choices

object[]

required

created
integer
required
The Unix timestamp (in seconds) of when the completion was created.

model
string
required
The model used for completion.

system_fingerprint
string
This fingerprint represents the backend configuration that the model runs with.

object
string
required
Possible values: [text_completion]

The object type, which is always "text_completion"

usage

object

curl
python
go
nodejs
ruby
csharp
php
java
powershell
OpenAI SDK
from openai import OpenAI

# user should set `base_url="https://api.deepseek.com/beta"` to use this feature.
client = OpenAI(
  api_key="<your API key>",
  base_url="https://api.deepseek.com/beta",
)
response = client.completions.create(
  model="deepseek-chat",
  prompt="def fib(a):",
  suffix="    return fib(a-1) + fib(a-2)",
  max_tokens=128)
print(response.choices[0].text)



REQUESTS
HTTP.CLIENT
import requests
import json

url = "https://api.deepseek.com/beta/completions"

payload = json.dumps({
  "model": "deepseek-chat",
  "prompt": "Once upon a time, ",
  "echo": False,
  "frequency_penalty": 0,
  "logprobs": 0,
  "max_tokens": 1024,
  "presence_penalty": 0,
  "stop": None,
  "stream": False,
  "stream_options": None,
  "suffix": None,
  "temperature": 1,
  "top_p": 1
})
headers = {
  'Content-Type': 'application/json',
  'Accept': 'application/json',
  'Authorization': 'Bearer <TOKEN>'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)


Request
Collapse all
Base URL
https://api.deepseek.com/beta
Auth
Bearer Token
Bearer Token
Body
 required
{
  "model": "deepseek-chat",
  "prompt": "Once upon a time, ",
  "echo": false,
  "frequency_penalty": 0,
  "logprobs": 0,
  "max_tokens": 1024,
  "presence_penalty": 0,
  "stop": null,
  "stream": false,
  "stream_options": null,
  "suffix": null,
  "temperature": 1,
  "top_p": 1
}
Send API Request
Response
Clear
Click the Send API Request button above and s


API ReferenceModelsLists Models
Lists Models
GET
https://api.deepseek.com/models
Lists the currently available models, and provides basic information about each one such as the owner and availability. Check Models & Pricing for our currently supported models.

Responses
200
OK, returns A list of models

application/json
Schema
Example (from schema)
Example
Schema

object
string
required
Possible values: [list]

data

Model[]

required

curl
python
go
nodejs
ruby
csharp
php
java
powershell
OpenAI SDK
from openai import OpenAI

# for backward compatibility, you can still use `https://api.deepseek.com/v1` as `base_url`.
client = OpenAI(api_key="<your API key>", base_url="https://api.deepseek.com")
print(client.models.list())



REQUESTS
HTTP.CLIENT
import requests

url = "https://api.deepseek.com/models"

payload={}
headers = {
  'Accept': 'application/json',
  'Authorization': 'Bearer <TOKEN>'
}

response = requests.request("GET", url, headers=headers, data=payload)

print(response.text)


Request
Collapse all
Base URL
https://api.deepseek.com
Auth
Bearer Token
Bearer Token
Send API Request
Response
Clear
Click the Send API Request button above and see the response here!


API ReferenceOthersGet User Balance
Get User Balance
GET
https://api.deepseek.com/user/balance
Get user current balance

Responses
200
OK, returns user balance info.

application/json
Schema
Example (from schema)
Example
Schema

is_available
boolean
Whether the user's balance is sufficient for API calls.

balance_infos

object[]

curl
python
go
nodejs
ruby
csharp
php
java
powershell
REQUESTS
HTTP.CLIENT
import requests

url = "https://api.deepseek.com/user/balance"

payload={}
headers = {
  'Accept': 'application/json',
  'Authorization': 'Bearer <TOKEN>'
}

response = requests.request("GET", url, headers=headers, data=payload)

print(response.text)


Request
Collapse all
Base URL
https://api.deepseek.com
Auth
Bearer Token
Bearer Token
Send API Request
Response
Clear
Click the Send API Request button above and see the response here!


Reasoning Model (deepseek-reasoner)
deepseek-reasoner is a reasoning model developed by DeepSeek. Before delivering the final answer, the model first generates a Chain of Thought (CoT) to enhance the accuracy of its responses. Our API provides users with access to the CoT content generated by deepseek-reasoner, enabling them to view, display, and distill it.

When using deepseek-reasoner, please upgrade the OpenAI SDK first to support the new input_schema.

pip3 install -U openai

API Input_schema
Input：

max_tokens：The maximum length of the final response after the CoT output is completed, defaulting to 4K, with a maximum of 8K. Note that the CoT output can reach up to 32K tokens, and the parameter to control the CoT length (reasoning_effort) will be available soon.
Output：

reasoning_content：The content of the CoT，which is at the same level as content in the output structure. See API Example for details
contentThe content of the final answer
Context Length：The API supports a maximum context length of 64K, and the length of the output reasoning_content is not counted within the 64K context length.

Supported Features：Chat Completion、Chat Prefix Completion (Beta)

Not Supported Features：Function Call、Json Output、FIM (Beta)

Not Supported Input_schema：temperature、top_p、presence_penalty、frequency_penalty、logprobs、top_logprobs. Please note that to ensure compatibility with existing software, setting temperature、top_p、presence_penalty、frequency_penalty will not trigger an error but will also have no effect. Setting logprobs、top_logprobs will trigger an error.

Multi-round Conversation
In each round of the conversation, the model outputs the CoT (reasoning_content) and the final answer (content). In the next round of the conversation, the CoT from previous rounds is not concatenated into the context, as illustrated in the following diagram:


Please note that if the reasoning_content field is included in the sequence of input messages, the API will return a 400 error. Therefore, you should remove the reasoning_content field from the API response before making the API request, as demonstrated in the API example.

API Example
The following code, using Python as an example, demonstrates how to access the CoT and the final answer, as well as how to conduct multi-round conversations:

NoStreaming
Streaming
from openai import OpenAI
client = OpenAI(api_key="<DeepSeek API Key>", base_url="https://api.deepseek.com")

# Round 1
messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]
response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=messages
)

reasoning_content = response.choices[0].message.reasoning_content
content = response.choices[0].message.content

# Round 2
messages.append({'role': 'assistant', 'content': content})
messages.append({'role': 'user', 'content': "How many Rs are there in the word 'strawberry'?"})
response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=messages
)
# ...

Previous
Get User Balance
Next
Multi-round Conversation
API Input_schema
Multi-round Conversation
API Example


from openai import OpenAI
client = OpenAI(api_key="<DeepSeek API Key>", base_url="https://api.deepseek.com")

# Round 1
messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]
response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=messages,
    stream=True
)

reasoning_content = ""
content = ""

for chunk in response:
    if chunk.choices[0].delta.reasoning_content:
        reasoning_content += chunk.choices[0].delta.reasoning_content
    else:
        content += chunk.choices[0].delta.content

# Round 2
messages.append({"role": "assistant", "content": content})
messages.append({'role': 'user', 'content': "How many Rs are there in the word 'strawberry'?"})
response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=messages,
    stream=True
)
# ...

Multi-round Conversation
This guide will introduce how to use the DeepSeek /chat/completions API for multi-turn conversations.

The DeepSeek /chat/completions API is a "stateless" API, meaning the server does not record the context of the user's requests. Therefore, the user must concatenate all previous conversation history and pass it to the chat API with each request.

The following code in Python demonstrates how to concatenate context to achieve multi-turn conversations.

from openai import OpenAI
client = OpenAI(api_key="<DeepSeek API Key>", base_url="https://api.deepseek.com")

# Round 1
messages = [{"role": "user", "content": "What's the highest mountain in the world?"}]
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages
)

messages.append(response.choices[0].message)
print(f"Messages Round 1: {messages}")

# Round 2
messages.append({"role": "user", "content": "What is the second?"})
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages
)

messages.append(response.choices[0].message)
print(f"Messages Round 2: {messages}")

In the first round of the request, the messages passed to the API are:

[
    {"role": "user", "content": "What's the highest mountain in the world?"}
]

In the second round of the request:

Add the model's output from the first round to the end of the messages.
Add the new question to the end of the messages.
The messages ultimately passed to the API are:

[
    {"role": "user", "content": "What's the highest mountain in the world?"},
    {"role": "assistant", "content": "The highest mountain in the world is Mount Everest."},
    {"role": "user", "content": "What is the second?"}
]


Chat Prefix Completion (Beta)
The chat prefix completion follows the Chat Completion API, where users provide an assistant's prefix message for the model to complete the rest of the message.

Notice
When using chat prefix completion, users must ensure that the role of the last message in the messages list is assistant and set the prefix parameter of the last message to True.
The user needs to set base_url="https://api.deepseek.com/beta" to enable the Beta feature.
Sample Code
Below is a complete Python code example for chat prefix completion. In this example, we set the prefix message of the assistant to "```python\n" to force the model to output Python code, and set the stop parameter to ['```'] to prevent additional explanations from the model.

from openai import OpenAI

client = OpenAI(
    api_key="<your api key>",
    base_url="https://api.deepseek.com/beta",
)

messages = [
    {"role": "user", "content": "Please write quick sort code"},
    {"role": "assistant", "content": "```python\n", "prefix": True}
]
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    stop=["```"],
)
print(response.choices[0].message.content)


FIM Completion (Beta)
In FIM (Fill In the Middle) completion, users can provide a prefix and a suffix (optional), and the model will complete the content in between. FIM is commonly used for content completion、code completion.

Notice
The max tokens of FIM completion is 4K.
The user needs to set base_url=https://api.deepseek.com/beta to enable the Beta feature.
Sample Code
Below is a complete Python code example for FIM completion. In this example, we provide the beginning and the end of a function to calculate the Fibonacci sequence, allowing the model to complete the content in the middle.

from openai import OpenAI

client = OpenAI(
    api_key="<your api key>",
    base_url="https://api.deepseek.com/beta",
)

response = client.completions.create(
    model="deepseek-chat",
    prompt="def fib(a):",
    suffix="    return fib(a-1) + fib(a-2)",
    max_tokens=128
)
print(response.choices[0].text)

Integration With Continue
Continue is a VSCode plugin that supports code completion. You can refer to this document to configure Continue for using the code completion feature.


JSON Output
In many scenarios, users need the model to output in strict JSON format to achieve structured output, facilitating subsequent parsing.

DeepSeek provides JSON Output to ensure the model outputs valid JSON strings.

Notice
To enable JSON Output, users should:

Set the response_format parameter to {'type': 'json_object'}.
Include the word "json" in the system or user prompt, and provide an example of the desired JSON format to guide the model in outputting valid JSON.
Set the max_tokens parameter reasonably to prevent the JSON string from being truncated midway.
When using the JSON Output feature, the API may occasionally return empty content. We are actively working on optimizing this issue. You can try modifying the prompt to mitigate such problems.
Sample Code
Here is the complete Python code demonstrating the use of JSON Output:

import json
from openai import OpenAI

client = OpenAI(
    api_key="<your api key>",
    base_url="https://api.deepseek.com",
)

system_prompt = """
The user will provide some exam text. Please parse the "question" and "answer" and output them in JSON format. 

EXAMPLE INPUT: 
Which is the highest mountain in the world? Mount Everest.

EXAMPLE JSON OUTPUT:
{
    "question": "Which is the highest mountain in the world?",
    "answer": "Mount Everest"
}
"""

user_prompt = "Which is the longest river in the world? The Nile River."

messages = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}]

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    response_format={
        'type': 'json_object'
    }
)

print(json.loads(response.choices[0].message.content))


The model will output:

{
    "question": "Which is the longest river in the world?",
    "answer": "The Nile River"
}


Function Calling
Function Calling allows the model to call external tools to enhance its capabilities.

Notice
The current version of the deepseek-chat model's Function Calling capabilitity is unstable, which may result in looped calls or empty responses. We are actively working on a fix, and it is expected to be resolved in the next version.

Sample Code
Here is an example of using Function Calling to get the current weather information of the user's location, demonstrated with complete Python code.

For the specific API format of Function Calling, please refer to the Chat Completion documentation.

from openai import OpenAI

def send_messages(messages):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=tools
    )
    return response.choices[0].message

client = OpenAI(
    api_key="<your api key>",
    base_url="https://api.deepseek.com",
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather of an location, the user shoud supply a location first",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"]
            },
        }
    },
]

messages = [{"role": "user", "content": "How's the weather in Hangzhou?"}]
message = send_messages(messages)
print(f"User>\t {messages[0]['content']}")

tool = message.tool_calls[0]
messages.append(message)

messages.append({"role": "tool", "tool_call_id": tool.id, "content": "24℃"})
message = send_messages(messages)
print(f"Model>\t {message.content}")

The execution flow of this example is as follows:

User: Asks about the current weather in Hangzhou
Model: Returns the function get_weather({location: 'Hangzhou'})
User: Calls the function get_weather({location: 'Hangzhou'}) and provides the result to the model
Model: Returns in natural language, "The current temperature in Hangzhou is 24°C."
Note: In the above code, the functionality of the get_weather function needs to be provided by the user. The model itself does not execute specific functions.


Context Caching
The DeepSeek API Context Caching on Disk Technology is enabled by default for all users, allowing them to benefit without needing to modify their code.

Each user request will trigger the construction of a hard disk cache. If subsequent requests have overlapping prefixes with previous requests, the overlapping part will only be fetched from the cache, which counts as a "cache hit."

Note: Between two requests, only the repeated prefix part can trigger a "cache hit." Please refer to the example below for more details.

Example 1: Long Text Q&A
First Request

messages: [
    {"role": "system", "content": "You are an experienced financial report analyst..."}
    {"role": "user", "content": "<financial report content>\n\nPlease summarize the key information of this financial report."}
]


Second Request

messages: [
    {"role": "system", "content": "You are an experienced financial report analyst..."}
    {"role": "user", "content": "<financial report content>\n\nPlease analyze the profitability of this financial report."}
]


In the above example, both requests have the same prefix, which is the system message + <financial report content> in the user message. During the second request, this prefix part will count as a "cache hit."

Example 2: Multi-round Conversation
First Request

messages: [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "What is the capital of China?"}
]

Second Request

messages: [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "What is the capital of China?"},
    {"role": "assistant", "content": "The capital of China is Beijing."},
    {"role": "user", "content": "What is the capital of the United States?"}
]

In this example, the second request can reuse the initial system message and user message from the first request, which will count as a "cache hit."

Example 3: Using Few-shot Learning
In practical applications, users can enhance the model's output performance through few-shot learning. Few-shot learning involves providing a few examples in the request to allow the model to learn a specific pattern. Since few-shot generally provides the same context prefix, the cost of few-shot is significantly reduced with the support of context caching.

First Request

messages: [    
    {"role": "system", "content": "You are a history expert. The user will provide a series of questions, and your answers should be concise and start with `Answer:`"},
    {"role": "user", "content": "In what year did Qin Shi Huang unify the six states?"},
    {"role": "assistant", "content": "Answer: 221 BC"},
    {"role": "user", "content": "Who was the founder of the Han Dynasty?"},
    {"role": "assistant", "content": "Answer: Liu Bang"},
    {"role": "user", "content": "Who was the last emperor of the Tang Dynasty?"},
    {"role": "assistant", "content": "Answer: Li Zhu"},
    {"role": "user", "content": "Who was the founding emperor of the Ming Dynasty?"},
    {"role": "assistant", "content": "Answer: Zhu Yuanzhang"},
    {"role": "user", "content": "Who was the founding emperor of the Qing Dynasty?"}
]


Second Request

messages: [    
    {"role": "system", "content": "You are a history expert. The user will provide a series of questions, and your answers should be concise and start with `Answer:`"},
    {"role": "user", "content": "In what year did Qin Shi Huang unify the six states?"},
    {"role": "assistant", "content": "Answer: 221 BC"},
    {"role": "user", "content": "Who was the founder of the Han Dynasty?"},
    {"role": "assistant", "content": "Answer: Liu Bang"},
    {"role": "user", "content": "Who was the last emperor of the Tang Dynasty?"},
    {"role": "assistant", "content": "Answer: Li Zhu"},
    {"role": "user", "content": "Who was the founding emperor of the Ming Dynasty?"},
    {"role": "assistant", "content": "Answer: Zhu Yuanzhang"},
    {"role": "user", "content": "When did the Shang Dynasty fall?"},        
]


In this example, 4-shots are used. The only difference between the two requests is the last question. The second request can reuse the content of the first 4 rounds of dialogue from the first request, which will count as a "cache hit."

Checking Cache Hit Status
In the response from the DeepSeek API, we have added two fields in the usage section to reflect the cache hit status of the request:

prompt_cache_hit_tokens: The number of tokens in the input of this request that resulted in a cache hit (0.1 yuan per million tokens).

prompt_cache_miss_tokens: The number of tokens in the input of this request that did not result in a cache hit (1 yuan per million tokens).

Hard Disk Cache and Output Randomness
The hard disk cache only matches the prefix part of the user's input. The output is still generated through computation and inference, and it is influenced by input_schema such as temperature, introducing randomness.

Additional Notes
The cache system uses 64 tokens as a storage unit; content less than 64 tokens will not be cached.

The cache system works on a "best-effort" basis and does not guarantee a 100% cache hit rate.

Cache construction takes seconds. Once the cache is no longer in use, it will be automatically cleared, usually within a few hours to a few days.


