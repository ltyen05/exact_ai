# EXACT 2026

## Cài đặt môi trường

```bash
pip install -r requirements.txt
```

## Chạy demo P1

Chạy demo logic:

```bash
python demo_p1.py --logic-samples 2 --physics-samples 0 --output outputs/demo_logic_p1_output.json
```

Chạy demo physics:

```bash
python demo_p1.py --logic-samples 0 --physics-samples 2 --output outputs/demo_physics_p1_output.json
```

Chạy cả hai phần:

```bash
python demo_p1.py --logic-samples 2 --physics-samples 2 --output outputs/demo_p1_output.json
```
