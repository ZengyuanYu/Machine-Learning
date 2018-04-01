def TimeTrains(ls):   # 处理时间格式
    lis = ls
    if len(ls) == 8 and '/' not in ls:
        year = ls[0:4]
        month = ls[4:6]
        day = ls[6:]
        lis = year + '-' + month + '-' + day
    elif len(ls) == 8 and '/' in ls:
        lss = ls.split('/')
        lis = lss[0] + '-' + '0' + lss[1] + '-' + '0' + lss[2]
    elif len(ls) == 6:
        if ls[0:2] != '20':
            year = '20' + ls[0:2]
            month = ls[2:4]
            day = ls[4:]
            lis = year + '-' + month + '-' + day
    elif len(ls) == 10 and '/' not in ls:
        lss = ls.split('.')
        lis = lss[0] + '-' + lss[1] + '-' + lss[2]
    elif len(ls) == 10 and '/' in ls:
        lss = ls.split('/')
        lis = lss[0] + '-' + lss[1] + '-' + lss[2]
    elif len(ls) == 7:
        year = ls[0:4]
        month = ls[4:5]
        day = ls[5:]
        lis = year + '-' + '0' + month + '-' + day
    elif '.' in ls:
        lss = ls.split('.')
        lis = lss[0] + '-' + lss[1] + '-' + '0' + lss[2]
    elif len(ls) == 9 and '/' not in ls:
        year = ls[0:4]
        month = ls[5:7]
        day = ls[7:]
        lis = year + '-' + month + '-' + day
    elif len(ls) == 9 and '/' in ls:
        lss = ls.split('/')
        lis = lss[0] + '-' + '0' + lss[1] + '-' + lss[2]
    return lis