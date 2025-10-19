# Utils Directory

보조 스크립트와 진단 도구 설명입니다.

- **terminate_uhd_claim.py**  
  - 이미 `claimed=True` 상태로 잡혀 있는 N3xx/N2xx 장치를 RPC로 해제합니다.  
  - 예: `python scripts/utils/terminate_uhd_claim.py --addr 192.168.10.1`  
  - UHD가 `Someone tried to claim this device again` 에러를 낼 때 사용.

- **packet_monitor.py**  
  - NIC 통계를 주기적으로 출력하거나 패킷 드롭을 확인할 수 있도록 만든 예제 스크립트입니다.  
  - `--interface`, `--interval` 옵션으로 감시 대상을 지정하세요.

추가 도구를 작성할 때는 이 디렉터리에 저장하고 README에 간단한 설명을 덧붙이면 됩니다.
