# === CASD-RM PATCH ===
import os as _cos
if _cos.environ.get('CASD_BANK'):
    from tau2.casd_rm_hook import install as _cinstall
    _cinstall(LLMAgent)
