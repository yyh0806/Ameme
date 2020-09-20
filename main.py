# @Time : 2020/9/5 10:54
# @Author : yangyuhui
# @Site : 
# @File : main.py
# @Software: PyCharm

from config import cfg
import logging
import sys
import os
import streamlit as st


def setup() -> object:
    # config
    cfg.merge_from_file("Ameme.yaml")
    cfg.freeze()
    # logging
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # UI
    st.sidebar.title(cfg.SIDEBAR.TITLE)
    selected_dataset = st.selectbox('Select', os.listdir(cfg.SIDEBAR.SUBHEADER_DATAPATH))
    return cfg


def main():
    config = setup()

    x = st.slider('x')  # 👈 this is a widget
    st.write(x, 'squared is', x * x)
    st.sidebar.selectbox('hhhh', ("email", "phone"))


if __name__ == "__main__":
    main()
