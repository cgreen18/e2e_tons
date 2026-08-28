

CHIPS_PER_BOARD = 4
BOARDS_PER_GROUP = 8
CHIPS_PER_GROUP = BOARDS_PER_GROUP * CHIPS_PER_BOARD
GROUPS = 32
TOTAL_CHIPS = GROUPS * CHIPS_PER_GROUP

def gbc_to_id(g,b,c):
    return g*CHIPS_PER_GROUP + b*CHIPS_PER_BOARD + c

def id_to_gbc(id):
    g = id // CHIPS_PER_GROUP
    b = (id % CHIPS_PER_GROUP) // CHIPS_PER_BOARD
    c = id % CHIPS_PER_BOARD
    return g,b,c

def add_intra_board_connections(G, chip_id):
    g,b,c = id_to_gbc(chip_id)
    for stride in range(1, CHIPS_PER_BOARD):
        target = gbc_to_id(g,b,(c+stride) % CHIPS_PER_BOARD)
        G.add_edge(chip_id, target)



def add_intra_group_connections(G, chip_id):
    g,b,c = id_to_gbc(chip_id)
    target = gbc_to_id(g,(b+c+1) % BOARDS_PER_GROUP, (c+2) % CHIPS_PER_BOARD)
    G.add_edge(chip_id, target)
    G.add_edge(target, chip_id)

def _decide_inter_group_connections(chip_id):
    group, rel = divmod(chip_id, CHIPS_PER_GROUP)

    if group != rel or FORCE_SYM:
        return ((rel*CHIPS_PER_GROUP) + group) % TOTAL_CHIPS

    other = (group + GROUPS//2) % % GROUPS
    return (other*CHIPS_PER_GROUP + far) % TOTAL_CHIPS

def add_inter_group_connections(G):
    for chip_id in range(TOTAL_CHIPS):
        target = _decide_inter_group_connections(chip_id)
        G.add_edge(chip_id, target)
        G.add_edge(target, chip_id)

def add_connections(G, chip_id):
    add_intra_board_connections(G, chip_id)
    add_intra_group_connections(G, chip_id)
    add_inter_group_connections(G, chip_id)