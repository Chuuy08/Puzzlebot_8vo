#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/spi/spidev.h>

#define DEVICE   "/dev/spidev1.0"
#define SPEED_HZ 1000000

#define CMD_STATUS 0xAA

#define STATE_IDLE               0x00
#define STATE_SUBIR              0x01
#define STATE_ESPERAR_ALINEACION 0x02
#define STATE_SUBIR_POCO         0x03
#define STATE_DELAY              0x04
#define STATE_BAJAR_VISION       0x05
#define STATE_ESPERAR_WAYPOINT   0x06
#define STATE_BAJAR_FINAL        0x07
#define STATE_FSM_DONE           0x08

static int fd;

int spi_init() {
    fd = open(DEVICE, O_RDWR);
    if (fd < 0) { perror("Error abriendo SPI"); return -1; }

    uint8_t  mode  = SPI_MODE_0;
    uint32_t speed = SPEED_HZ;
    uint8_t  bits  = 8;

    if (ioctl(fd, SPI_IOC_WR_MODE,          &mode)  < 0) { perror("mode");  return -1; }
    if (ioctl(fd, SPI_IOC_WR_MAX_SPEED_HZ,  &speed) < 0) { perror("speed"); return -1; }
    if (ioctl(fd, SPI_IOC_WR_BITS_PER_WORD, &bits)  < 0) { perror("bits");  return -1; }

    printf("SPI inicializado en %s\n", DEVICE);
    return 0;
}

int spi_transfer(uint8_t *tx, uint8_t *rx, int len) {
    struct spi_ioc_transfer tr = {
        .tx_buf        = (unsigned long)tx,
        .rx_buf        = (unsigned long)rx,
        .len           = len,
        .speed_hz      = SPEED_HZ,
        .bits_per_word = 8,
        .delay_usecs   = 0,
    };
    if (ioctl(fd, SPI_IOC_MESSAGE(1), &tr) < 0) {
        perror("Error en transferencia SPI");
        return -1;
    }
    return 0;
}

uint8_t spi_get_estado() {
    uint8_t tx = CMD_STATUS;
    uint8_t rx = 0;
    spi_transfer(&tx, &rx, 1);
    return rx;
}

int esperar_estado(uint8_t estado_esperado, int timeout_intentos) {
    uint8_t actual;
    const char *nombres[] = {
        "IDLE", "SUBIR", "ESPERAR_ALINEACION",
        "SUBIR_POCO", "DELAY", "BAJAR_VISION",
        "ESPERAR_WAYPOINT", "BAJAR_FINAL", "FSM_DONE"
    };

    for (int i = 0; i < timeout_intentos; i++) {
        actual = spi_get_estado();
        if (actual <= 8)
            printf("  [%d] Estado FPGA: %s (0x%02X)\n", i, nombres[actual], actual);
        else
            printf("  [%d] Estado FPGA: 0x%02X\n", i, actual);

        if (actual == estado_esperado) return 1;
        usleep(100000);
    }
    printf("  TIMEOUT\n");
    return 0;
}

void spi_send_secuencia(uint8_t  cmd,
                        uint16_t t_subir,
                        uint16_t t_subir_poco,
                        uint16_t t_delay,
                        uint16_t t_bajar_vision,
                        uint16_t t_bajar_final) {
    uint8_t tx[11] = {
        cmd,
        (t_subir >> 8),        (t_subir & 0xFF),
        (t_subir_poco >> 8),   (t_subir_poco & 0xFF),
        (t_delay >> 8),        (t_delay & 0xFF),
        (t_bajar_vision >> 8), (t_bajar_vision & 0xFF),
        (t_bajar_final >> 8),  (t_bajar_final & 0xFF),
    };
    uint8_t rx[11] = {0};
    spi_transfer(tx, rx, 11);
    printf("Secuencia: cmd=0x%02X subir=%dms subir_poco=%dms "
           "delay=%dms bajar_vision=%dms bajar_final=%dms\n",
           cmd, t_subir, t_subir_poco, t_delay, t_bajar_vision, t_bajar_final);
}

void spi_send_command(uint8_t cmd) {
    uint8_t tx = cmd;
    uint8_t rx = 0;
    spi_transfer(&tx, &rx, 1);
    printf("Comando enviado: 0x%02X\n", cmd);
}

void spi_close() {
    close(fd);
}
