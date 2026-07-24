"""Quantitative Crypto Portfolio Risk Framework.

Lightweight package init — does **not** import data-source modules
(``coingecko_client``, ``yfinance_client``, ``data_loader``) so importing
this package never pulls in optional dependencies like ``yfinance`` or
network clients.

Import submodules explicitly, for example::

    from var_cvar_crypto_risk.var_models import calculate_var
    from var_cvar_crypto_risk.backtesting import backtest_var_model
"""

__version__ = "1.0.0"
__project__ = "Quantitative Crypto Portfolio Risk Framework"
