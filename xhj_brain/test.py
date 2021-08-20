from map import mapping
from inference import run_inf

# 输入用户属性，预测高维房源值
# user_lst = ['CostFrom','CostTo','Room','Sex',]
a = run_inf([3000, 4000, 3, "恒大绿洲"])
print('user  -----> house : {}'.format(a))

# 输入房源属性，输出高维房源值
# item_lst = ['BuildingSize','FloorNumber','RoomCount','HallCount','ToiletCount',
#             'Price','TotalPrice','EstateName','City_Name','Sq_Name']
b = mapping([100, 5, 3, 2, 1, 5000, 50, "红星商务楼", "长沙", "南门口"])
print('house -----> house : {}'.format(b))