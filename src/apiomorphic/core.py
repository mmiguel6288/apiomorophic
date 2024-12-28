import re
import json
from copy import deepcopy
from typing import Dict, List, Union, Optional, Any


class FromBase:
    pass
class ToBase:
    pass

def translate(source : str,target : str) -> ToBase:
    """Translate between API formats.
    
    Args:
        source: Source API format ('openai' or 'anthropic')
        target: Target API format ('openai' or 'anthropic')
        
    Returns:
        ToBase: Appropriate converter class
        
    Raises:
        Exception: If invalid source/target pair
    """
    match (source, target):
        case ('openai','anthropic'):
            return FromOpenAi.ToAnthropic
        case ('anthropic','openai'):
            return FromAnthropic.ToOpenAi
        case _:
            raise Exception(f'Invalid (source,target) pair: ({source},{target})')

class FromAnthropic(FromBase):
    class ToOpenAi(ToBase):
        @staticmethod
        def convert_tool_schema(tool_schema_entry : Dict[str, Any], strict : bool = False):
            #anthropic
            # {
            #     ... ,
            #     'tools':[
            #         {
            #             'name': name,
            #             'description': description,
            #             'input_schema': json_schema,
            #         },
            #         ... ,
            #         ],
            # }

            #openai:
            # {
            #     ... ,
            #     'tools':[
            #         {
            #             'type':'function',
            #             'function':{
            #                 'description': description,
            #                 'name': name,
            #                 'parameters':json_schema,
            #                 'strict': boolean_strict_schema_adherence,
            #                 },
            #         },
            #         ... ,
            #       ]
            #     }
            # }
            result = {
                    'type':'function',
                    'function':{
                        'name': tool_schema_entry['name'],
                        'strict' : strict ,
                        }
                    },
            if 'description' in tool_schema_entry:
                result['function']['description'] = tool_schema_entry['description']

            if 'input_schema' in tool_schema_entry:
                result['function']['parameters'] = tool_schema_entry['input_schema']
            else:
                result['function']['parameters'] = {'type':'object','properties':{}}
            return result

        @classmethod
        def convert_message(cls,msg : Dict[str,Any] ,image_detail : str ='auto') -> Dict[str,Any]:
            #https://platform.openai.com/docs/api-reference/chat/create
            #https://docs.anthropic.com/en/api/messages
            output_messages = []
            match msg['role']:
                case 'assistant': 
                    if 'content' in msg and isinstance(msg['content'],list):
                        for entry in msg['content']:
                            match entry['type']:
                                case 'tool_use':
                                    #tool request anthropic:
                                    #{"role":"assistant","content":[
                                    #   ...,
                                    #   {"type":"tool_use",
                                    #   "id":...,
                                    #   "name":...,
                                    #   "input":...,}
                                    #   ...,
                                    #   ]}

                                    #tool request openai:
                                    #{"role":"assistant",
                                    # "tool_calls":[{
                                    #   "id":...,
                                    #   "type":"function",
                                    #   "function":{
                                    #       "name":...,
                                    #       "arguments":...,
                                    #       }
                                    #   },
                                    #   ...,
                                    #]}

                                    output_messages.append({
                                        'role':'assistant',
                                        'tool_calls':[
                                            {
                                                'id':entry['id'],
                                                'type':'function',
                                                'function':{
                                                    'name':entry['name'],
                                                    'arguments':json.dumps(entry['input'])
                                                    }
                                                }
                                            ]
                                        });
                                case 'text':
                                    output_messages.append({'role':'assistant','content':entry['text']})
                                case _:
                                    breakpoint()
                    else:
                        output_messages.append(deepcopy(msg))
                case 'user':
                    if 'content' in msg and isinstance(msg['content'],list):
                        new_msg = None
                        for entry in msg['content']:
                            match entry['type']:
                                case 'tool_result':
                                    #tool response anthropic:
                                    #{"role":"user",
                                    #"content":[
                                    #   {
                                    #       "type":"tool_result",
                                    #       "tool_use_id":...,
                                    #       "content":...,
                                    #   }
                                    #   ...
                                    #   ]}

                                    #tool response openai:
                                    #{"role":"tool",
                                    #   "tool_call_id":...
                                    #   "content":...,
                                    #}
                                    if new_msg is not None:
                                        output_messages.append(new_msg)
                                        new_msg = None
                                    output_messages.append({
                                        'role':'tool',
                                        'tool_call_id':entry['tool_use_id'],
                                        'content':entry['content']
                                        })
                                case 'text':
                                    if new_msg is None:
                                        new_msg = {'role':'user','content':[]}
                                    new_msg['content'].append({'type':'text','text':entry['text']})
                                case 'image':
                                    # anthropic
                                    #   {
                                    #       'role':'user',
                                    #       'content':[
                                    #           {
                                    #               'type':'image',
                                    #               'source': {
                                    #                   'type':'base64',
                                    #                   'media_type':f'image/{image_format}',
                                    #                   'data':image_data,
                                    #               }
                                    #           }
                                    #       ]
                                    #   }

                                    # openai:
                                    #   {
                                    #       'role':'user',
                                    #       'content':[
                                    #           {
                                    #               "type": "image_url",
                                    #               "image_url": {
                                    #                   "url": f"data:image/{image_format};base64,{image_data}",
                                    #                   "detail": image_detail
                                    #               }
                                    #           }
                                    #   }
                                    image_format = re.search('^image/(.+$)',entry['source']['media_type']).group(1)
                                    if new_msg is None:
                                        new_msg = {'role':'user','content':[]}
                                    image_data = entry['source']['data']
                                    new_msg['content'].append({
                                        'type':'image_url',
                                        'image_url': {
                                            'url': f'data:image/{image_format};base64,{image_data}',
                                            'detail':image_detail,
                                            }
                                        })
                                case _:
                                    breakpoint()
                        if new_msg is not None:
                            output_messages.append(new_msg)
                    else:
                        output_messages.append(deepcopy(msg))
                case _:
                    output_messages.append(deepcopy(msg))
            return output_messages
        @classmethod
        def convert(cls,api_params : Dict[str,Any]) -> Dict[str, Any]:
            new_params = deepcopy(api_params)
            messages = []
            for msg in new_params['messages']:
                messages.extend(cls.convert_message(msg))
            new_params['messages'] = messages
            return new_params

class FromOpenAi(FromBase):
    class ToAnthropic(ToBase):

        @staticmethod
        def convert_tool_schema(tool_schema_entry : Dict[str,Any]) -> Dict[str,Any]:
            #openai:
            # {
            #     ... ,
            #     'tools':[
            #         {
            #             'type':'function',
            #             'function':{
            #                 'description': description,
            #                 'name': name,
            #                 'parameters':json_schema,
            #                 'strict': boolean_strict_schema_adherence,
            #                 },
            #         },
            #         ... ,
            #       ]
            #     }
            # }

            #anthropic
            # {
            #     ... ,
            #     'tools':[
            #         {
            #             'name': name,
            #             'description': description,
            #             'input_schema': json_schema,
            #         },
            #         ... ,
            #         ],
            # }


            result = {
                    'name':tool_schema_entry['function']['name'],
                    }
            if 'description' in tool_schema_entry['function']:
                result['description'] = tool_schema_entry['function']['description']

            if 'parameters' in tool_schema_entry['function']:
                result['input_schema'] = tool_schema_entry['function']['parameters']
            else:
                result['input_schema'] = {'type':'object','properties':{}}
            return result


        @staticmethod
        def convert_vision(msg : Dict[str, Any]) -> Dict[str,Any]:
            # openai:
            # {
            #     "type": "image_url",
            #     "image_url": {
            #         "url": f"data:image/{image_format};base64,{image_data}",
            #         "detail": image_detail
            #     }
            # 
            # anthropic
            # {'type':'image',
            # 'source': {
            #     'type':'base64',
            #     'media_type':f'image/{image_format}',
            #     'data':image_data,
            #     }
            # }
            if 'content' in msg and isinstance(msg['content'],list):
                content = []
                for content_entry in msg['content']:
                    if content_entry['type'] == 'image_url':

                        image_format,image_data = re.search(
                                '^data:image/([^;]+);base64,(.+$)',
                                content_entry['image_url']['url']
                                ).groups()

                        content.append({
                            'type':'image',
                            'source': {
                                'type':'base64',
                                'media_type':f'image/{image_format}',
                                'data':image_data,
                                },
                            })
                    else:
                        content.append(content_entry)
                msg['content'] = content
            return msg


        @classmethod
        def convert_message(cls,msg : Dict[str,Any]) -> Dict[str,Any]:
            #https://platform.openai.com/docs/api-reference/chat/create
            #https://docs.anthropic.com/en/api/messages
            output_messages = []
            match msg['role']:
                case 'assistant': 
                    if 'tool_calls' in msg:
                        #tool request openai:
                        #{"role":"assistant",
                        # "tool_calls":[{
                        #   "id":...,
                        #   "type":"function",
                        #   "function":{
                        #       "name":...,
                        #       "arguments":...,
                        #       }
                        #   },
                        #   ...,
                        #]}

                        #tool request anthropic:
                        #{
                        #   "role":"assistant",
                        #   "content":[
                        #       ...,
                        #       {"type":"tool_use",
                        #       "id":...,
                        #       "name":...,
                        #       "input":...,}
                        #       ...,
                        #   ]}
                        for tool_call_entry in msg['tool_calls']:
                            output_messages.append({
                                'role':'assistant',
                                'content':[
                                    {
                                        'type':'tool_use',
                                        'id':tool_call_entry['id'],
                                        'name':tool_call_entry['function']['name'],
                                        'input':json.loads(tool_call_entry['function']['arguments']),
                                        }
                                    ]
                                })
                    else:
                        output_messages.append(msg)
                case 'tool':
                    #tool response openai:
                    #{"role":"tool",
                    #   "tool_call_id":...
                    #   "content":...,
                    #}

                    #tool response anthropic:
                    #{"role":"user",
                    #"content":[
                    #   {
                    #       "type":"tool_result",
                    #       "tool_use_id":...,
                    #       "content":...,
                    #   }
                    #   ...
                    #   ]}

                    output_messages.append({
                        'role':'user',
                        'content':[{
                            'type':'tool_result',
                            'tool_use_id':msg['tool_call_id'],
                            'content':msg['content']}]
                        })
                case 'system':
                    output_messages.append(msg)
                case 'user':
                    output_messages.append(cls.convert_vision(msg))
            return output_messages


        @classmethod
        def convert(cls,api_params : Dict[str,Any]) -> Dict[str,Any]:
            new_params = deepcopy(api_params)
            # misc params
            n = new_params.get('n')
            if n is not None and n != 1:
                raise Exception('Anthropic API only supports n=1')
            if 'stream' in new_params:
                del new_params['stream']
            # if 'max_tokens' not in new_params:
            #     new_params['max_tokens'] = 1024

            #separate system messages
            messages = new_params['messages']
            system_messages = []
            other_messages = []
            for msg in messages:
                #can a system prompt have an image? 
                if msg['role'] == 'system':
                    if isinstance(msg['content'],list):
                        system_messages.extend([entry['text'] for entry in msg['content']])
                    else:
                        system_messages.append(msg['content'])
                else:
                    other_messages.extend(cls.convert_message(msg))
            system_message = '\n'.join(system_messages).strip() if system_messages else ''
            new_params['messages'] = other_messages
            if len(system_message) > 0:
                new_params['system'] = system_message


            #convert tool schema
            if 'tools' in new_params:
                new_params['tools'] = [cls.convert_tool_schema(tool_schema_entry) for tool_schema_entry in new_params['tools']]
            return new_params
