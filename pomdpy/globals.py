import numpy as np

bt_global = None


def get_belief_state(bn):
    pt_filter = bn.state_particles
    freq_dict = {}

    for pt in pt_filter:
        # print(pt.rock_states)
        if ((pt.position.to_string(), tuple(pt.rock_states)) not in freq_dict.keys()):
            freq_dict[(pt.position.to_string(), tuple(pt.rock_states))] = 1
        else:
            freq_dict[(pt.position.to_string(), tuple(pt.rock_states))] += 1

    belief_state = np.zeros(2 + len(pt_filter[0].rock_states))
    for key in freq_dict:
        fraction = freq_dict[key]/len(pt_filter)

        # parse position from the string
        cleaned_string = key[0].strip('()')
        posx, posy = cleaned_string.split(',')
        posx = int(posx)
        posy = int(posy)

        curr_state_list = [posx, posy] + list(key[1])
        curr_state_ar = np.array(curr_state_list)
        belief_state += fraction * curr_state_ar

    return belief_state


def get_qvals(bn):
    mapping = bn.action_map
    actions = list(mapping.entries.values())
    qval = []
    for action_entry in actions:
        current_q = action_entry.mean_q_value
        qval.append(current_q)
    return np.array(qval)
