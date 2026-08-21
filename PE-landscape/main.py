import os  
from input_reader import read_input_file  
from flow_control import main_flow_control  
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
  
def run():  
    input_file = 'input.txt'  
    params = read_input_file(input_file)  
    main_flow_control(params)  
  
if __name__ == "__main__":  
    run()
