#prabhsi
import json
import torch
import torch.nn.functional as F
import re

#this verifies whether the mcq json is valid or not
# def parse_mcq(output_text):
#     try:
#         mcq = json.loads(output_text)

#         assert "question" in mcq
#         assert "choices" in mcq
#         assert "answer" in mcq

#         assert isinstance(mcq["choices"], list)
#         assert len(mcq["choices"]) == 4
#         assert mcq["answer"] in ["A", "B", "C", "D"]

#         return mcq
#     except Exception:
#         return None

def parse_mcq(output_text):
    try:
        mcq = json.loads(output_text)

        required_keys = ["topic", "question", "choices", "answer"]
        for key in required_keys:
            if key not in mcq:
                return None

        if not isinstance(mcq["topic"], str):
            return None
        if not isinstance(mcq["question"], str):
            return None
        # if not isinstance(mcq["explanation"], str):
        #     return None

        if not isinstance(mcq["choices"], list) or len(mcq["choices"]) != 4:
            return None

        expected_labels = ["A)", "B)", "C)", "D)"]
        for choice, label in zip(mcq["choices"], expected_labels):
            if not isinstance(choice, str):
                return None
            if not choice.strip().startswith(label):
                return None

        if mcq["answer"] not in ["A", "B", "C", "D"]:
            return None

        if len(mcq["explanation"].split()) > 100:
            return None

        return mcq

    except Exception:
        return None


def build_prompt(mcq_dict):
    return json.dumps(mcq_dict, indent=2)

def get_prediction(agent, mcq_text):
    response, _, _ = agent.generate_response(
        mcq_text,
        temperature=0.0,
        do_sample=False,
        max_new_tokens=5,
    )
    return response.strip()

def get_letter_prob(agent, prompt_text):
    tokenizer = agent.tokenizer
    model = agent.model

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)

    letters = ["A", "B", "C", "D"]
    token_ids = [
        tokenizer.encode(x, add_special_tokens=False)[0]
        for x in letters
    ]

    selected_probs = probs[0, token_ids]
    return dict(zip(letters, selected_probs))

"""
#a-agents rewards is the log prob of the correct answer
def compute_a_reward(a_agent, mcq_json):
    mcq = parse_mcq(mcq_json)
    if mcq is None:
        return -10.0

    correct = mcq["answer"]
    prompt_text = build_prompt(mcq)

    tokenizer = a_agent.tokenizer
    model = a_agent.model

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)

    token_id = tokenizer.encode(correct, add_special_tokens=False)[0]
    p_correct = probs[0, token_id]

    return torch.log(p_correct + 1e-8).item()

#this is diff
#this outputs 1.0 for correct pred, -1.0 for incorrect pred, -10 for invalid mcq json
def compute_a_agent_normal_reward(a_agent, mcq_json):
    mcq = parse_mcq(mcq_json)
    if mcq is None:
        return -10.0

    correct = mcq["answer"].strip()
    prompt_text = build_prompt(mcq)

    tokenizer = a_agent.tokenizer
    model = a_agent.model

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, do_sample=False)

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    predicted = generated_text.split()[0].strip()

    if predicted == correct:
        retrurn 1.0
    else:
        return -1.0

"""
#entropy based reward
#-10.0 for invalid json
#-2.0 for incorrect pred
def compute_q_reward(
    q_output,
    a_agent,
    baseline_agent=None,
):
    mcq = parse_mcq(q_output)

    if mcq is None:
        return -10.0

    correct = mcq["answer"]
    prompt_text = build_prompt(mcq)

    our_pred = get_prediction(a_agent, prompt_text)

    if our_pred != correct:
        return -2.0

    reward = 1.0 

    if baseline_agent is not None:
        base_pred = get_prediction(baseline_agent, prompt_text)
        if base_pred != correct:
            reward += 2.0

        entropy = get_entropy_of_answer_distribution(
            baseline_agent, prompt_text
        )
        reward += 0.5 * entropy

    return reward

#TRL version reward function for A-Agent
# def a_agent_reward_fn(samples, model=None, tokenizer=None, **kwargs):
#     rewards = []
#     prompts = kwargs.get("prompts")

#     for output, prompt in zip(samples, prompts):
#         mcq = parse_mcq(prompt)

#         if mcq is None:
#             # rewards.append(-5.0) -> not needed, only to be done on q-agent side
#             continue

#         correct = mcq["answer"]
#         predicted = output.strip()

#         if predicted != correct:
#             rewards.append(-2.0)
#         else:
#             rewards.append(1.0)

#     return rewards

def a_agent_reward_fn(prompts, completions, **kwargs):

    rewards = []

    for prompt, output in zip(prompts, completions):
        mcq = parse_mcq(prompt)

        if mcq is None:
            # rewards.append(-5.0)
            continue

        correct = mcq["answer"]
        pred = output.strip()

        if pred == correct:
            rewards.append(1.0)
        else:
            rewards.append(-2.0)

    return rewards



#TRL version reward function for Q-Agent
def build_q_reward_fn(a_agent, baseline_agent=None):
    def q_agent_reward_fn(samples, **kwargs):
        rewards = []

        for sample in samples:
            mcq = parse_mcq(sample)

            if mcq is None:
                rewards.append(-10.0)
                continue

            correct = mcq["answer"]
            prompt_text = build_prompt(mcq)

            #a-agent pred
            our_pred = get_prediction(a_agent, prompt_text)
            #need to ponder over this, should q-agent get -ve reward if a-agent gives wrong answer? how to judge difficulty?
            # Q -> A , AG -> A
            # predicted != expected 
            if our_pred != correct:
                rewards.append(-2.0)
                continue

            reward = 1.0

            #adverserial baseline pred
            #baseline helps judge difficulty
            if baseline_agent is not None:
                base_pred = get_prediction(
                    baseline_agent, prompt_text
                )

                if base_pred != correct:
                    reward += 2.0

                probs = get_letter_prob(
                    baseline_agent, prompt_text
                )

                # entropy = -sum(
                #     p * torch.log(p + 1e-8)
                #     for p in probs.values()
                # )
                # reward += 0.5 * entropy.item()

                probs_tensor = torch.stack(list(probs.values()))
                entropy = -torch.sum(probs_tensor * torch.log(probs_tensor + 1e-8))
                reward += 0.5 * entropy.item()


            rewards.append(float(reward))

        return rewards

    return q_agent_reward_fn