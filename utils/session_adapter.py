import streamlit as st


class SessionAdapter:
    """
    A clean wrapper for Streamlit session_state.
    Provides namespacing, safer access, and testability.
    """

    def __init__(self, namespace: str):
        self.namespace = namespace

    def _full_key(self, key: str) -> str:
        return f"{self.namespace}_{key}"

    def get(self, key: str, default=None):
        return st.session_state.get(self._full_key(key), default)

    def set(self, key: str, value):
        st.session_state[self._full_key(key)] = value
