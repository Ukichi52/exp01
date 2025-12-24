import csv
import json
import sys
from pathlib import Path

def csv_to_jsonl(input_csv_path, output_jsonl_path=None):
    """
    将CSV文件转换为JSONL格式
    
    Args:
        input_csv_path (str): 输入CSV文件路径
        output_jsonl_path (str, optional): 输出JSONL文件路径。如果为None，则使用输入文件名
    """
    # 如果未指定输出路径，则使用输入文件名（修改扩展名）
    if output_jsonl_path is None:
        input_path = Path(input_csv_path)
        output_jsonl_path = input_path.with_suffix('.jsonl')
    
    try:
        # 读取CSV文件
        with open(input_csv_path, 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            
            # 写入JSONL文件
            with open(output_jsonl_path, 'w', encoding='utf-8') as jsonl_file:
                for row in csv_reader:
                    # 将每行转换为JSON格式并写入
                    json_line = json.dumps(row, ensure_ascii=False)
                    jsonl_file.write(json_line + '\n')
        
        print(f"转换成功！")
        print(f"输入文件: {input_csv_path}")
        print(f"输出文件: {output_jsonl_path}")
        print(f"转换行数: 请查看输出文件的行数")
        
        # 验证转换结果
        with open(output_jsonl_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"实际转换行数: {len(lines)}")
        
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_csv_path}")
    except Exception as e:
        print(f"转换过程中发生错误: {str(e)}")

def main():
    """
    主函数：处理命令行参数或交互式输入
    """
    # 如果有命令行参数
    if len(sys.argv) > 1:
        input_csv = sys.argv[1]
        
        # 如果有第二个参数，则作为输出文件路径
        output_jsonl = sys.argv[2] if len(sys.argv) > 2 else None
        
        csv_to_jsonl(input_csv, output_jsonl)
    else:
        # 交互式模式
        print("=== CSV转JSONL转换器 ===")
        input_csv = input("请输入CSV文件路径: ").strip()
        
        if not input_csv:
            print("未提供输入文件路径，程序退出。")
            return
        
        output_option = input("是否指定输出文件路径？(y/n，默认n): ").strip().lower()
        
        if output_option == 'y':
            output_jsonl = input("请输入JSONL输出文件路径: ").strip()
            csv_to_jsonl(input_csv, output_jsonl)
        else:
            csv_to_jsonl(input_csv)

if __name__ == "__main__":
    main()