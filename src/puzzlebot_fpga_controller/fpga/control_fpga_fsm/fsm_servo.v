module spi_motor_top (
    input  wire clk,
    input  wire rst_n,
    input  wire SCK,
    input  wire MOSI,
    output wire MISO,
    input  wire CS,
    output reg  servo_pwm,
    output reg  led_debug
);

// ═══════════════════════════════════════════════
// SINCRONIZACIÓN SPI
// ═══════════════════════════════════════════════
reg [1:0] SCK_r;  always @(posedge clk) SCK_r  <= {SCK_r[0],  SCK };
reg [2:0] CS_r;   always @(posedge clk) CS_r   <= {CS_r[1:0], CS  };
reg [1:0] MOSI_r; always @(posedge clk) MOSI_r <= {MOSI_r[0], MOSI};

wire SCK_rising  = (SCK_r[1:0] == 2'b01);
wire SCK_falling = (SCK_r[1:0] == 2'b10);
wire CS_falling  = (CS_r[2:1]  == 2'b10);
wire CS_active   = ~CS_r[1];
wire MOSI_sync   =  MOSI_r[1];

// ═══════════════════════════════════════════════
// RECEPTOR SPI
// Mensaje inicial : 11 bytes (cmd + 5 tiempos)
// Mensajes continuación: 1 byte
// ═══════════════════════════════════════════════
reg [2:0]  bit_cnt    = 0;
reg [3:0]  byte_cnt   = 0;
reg [7:0]  rx_buf     = 0;
reg [7:0]  rx_byte    = 0;
reg        byte_ready = 0;
reg        valid      = 0;

reg [7:0]  cmd_reg            = 0;
reg [15:0] t_subir_reg        = 0;
reg [15:0] t_subir_poco_reg   = 0;
reg [15:0] t_delay_reg        = 0;
reg [15:0] t_bajar_vision_reg = 0;
reg [15:0] t_bajar_final_reg  = 0;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        bit_cnt               <= 0;
        byte_cnt              <= 0;
        rx_buf                <= 0;
        rx_byte               <= 0;
        byte_ready            <= 0;
        valid                 <= 0;
        cmd_reg               <= 0;
        t_subir_reg           <= 0;
        t_subir_poco_reg      <= 0;
        t_delay_reg           <= 0;
        t_bajar_vision_reg    <= 0;
        t_bajar_final_reg     <= 0;
    end else begin
        byte_ready <= 0;
        valid      <= 0;

        if (CS_falling) begin
            bit_cnt  <= 0;
            byte_cnt <= 0;
            rx_buf   <= 0;
        end

        if (CS_active && SCK_rising) begin
            rx_buf <= {rx_buf[6:0], MOSI_sync};

            if (bit_cnt == 3'd7) begin
                rx_byte    <= {rx_buf[6:0], MOSI_sync};
                byte_ready <= 1;
                bit_cnt    <= 3'd0;
                begin : inc_byte
                    reg [4:0] byte_next;
                    byte_next = {1'b0, byte_cnt} + 5'd1;
                    byte_cnt <= byte_next[3:0];
                end
            end else begin
                begin : inc_bit
                    reg [3:0] bit_next;
                    bit_next = {1'b0, bit_cnt} + 4'd1;
                    bit_cnt <= bit_next[2:0];
                end
            end
        end

        if (byte_ready) begin
            case (byte_cnt - 1)
                4'd0:  cmd_reg                  <= rx_byte;
                4'd1:  t_subir_reg[15:8]        <= rx_byte;
                4'd2:  t_subir_reg[7:0]         <= rx_byte;
                4'd3:  t_subir_poco_reg[15:8]   <= rx_byte;
                4'd4:  t_subir_poco_reg[7:0]    <= rx_byte;
                4'd5:  t_delay_reg[15:8]        <= rx_byte;
                4'd6:  t_delay_reg[7:0]         <= rx_byte;
                4'd7:  t_bajar_vision_reg[15:8] <= rx_byte;
                4'd8:  t_bajar_vision_reg[7:0]  <= rx_byte;
                4'd9:  t_bajar_final_reg[15:8]  <= rx_byte;
                4'd10: begin
                    t_bajar_final_reg[7:0] <= rx_byte;
                    valid <= 1;
                end
            endcase
        end
    end
end

// ═══════════════════════════════════════════════
// TRANSMISOR MISO
// Responde con estado cuando recibe 0xAA
// Responde con 0xFF cuando FSM termina
// ═══════════════════════════════════════════════
reg [7:0] tx_shift    = 8'h00;
reg       ack_pending = 0;

wire fsm_done_pulse;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        ack_pending <= 0;
        tx_shift    <= 8'h00;
    end else begin
        // FSM terminó — marcar ACK pendiente
        if (fsm_done_pulse)
            ack_pending <= 1;

        // CS baja → cargar byte a transmitir
        if (CS_falling) begin
            if (rx_byte == 8'hAA)
                // Petición de estado — responde con estado actual [3:0]
                tx_shift <= {4'b0000, state};
            else if (ack_pending) begin
                tx_shift    <= 8'hFF;  // ACK de secuencia completada
                ack_pending <= 0;
            end else
                tx_shift <= 8'h00;
        end

        // Flanco bajada SCK → desplazar bit
        if (CS_active && SCK_falling)
            tx_shift <= {tx_shift[6:0], 1'b0};
    end
end

assign MISO = CS_active ? tx_shift[7] : 1'bz;

// ═══════════════════════════════════════════════
// GENERADOR PWM — 20ms a 27MHz
// ═══════════════════════════════════════════════
localparam PERIOD     = 21'd1_000_000;
localparam PULSE_SUBE = 20'd56_000;
localparam PULSE_STOP = 20'd54_000;
localparam PULSE_BAJA = 20'd53_000;

reg [19:0] pwm_counter = 0;
reg [19:0] pulse       = PULSE_STOP;

always @(posedge clk) begin
    if (pwm_counter == PERIOD - 1) begin
        pwm_counter <= 20'd0;
    end else begin
        begin : inc_pwm
            reg [20:0] pwm_next;
            pwm_next = {1'b0, pwm_counter} + 21'd1;
            pwm_counter <= pwm_next[19:0];
        end
    end
end

always @(posedge clk)
    servo_pwm <= (pwm_counter < pulse);

// ═══════════════════════════════════════════════
// FSM MOTOR
// ═══════════════════════════════════════════════
localparam CICLOS_POR_MS = 32'd27_000;

localparam CMD_RACK     = 8'h10;
localparam CMD_RODILLO  = 8'h11;
localparam CMD_ALINEADO = 8'h21;
localparam CMD_WAYPOINT = 8'h22;

localparam IDLE               = 4'd0;
localparam SUBIR              = 4'd1;
localparam ESPERAR_ALINEACION = 4'd2;
localparam SUBIR_POCO         = 4'd3;
localparam DELAY              = 4'd4;
localparam BAJAR_VISION       = 4'd5;
localparam ESPERAR_WAYPOINT   = 4'd6;
localparam BAJAR_FINAL        = 4'd7;
localparam FSM_DONE           = 4'd8;

reg [3:0]  state   = IDLE;
reg [31:0] counter = 0;
reg [31:0] target  = 0;
reg        done_r  = 0;

reg [15:0] r_t_subir;
reg [15:0] r_t_subir_poco;
reg [15:0] r_t_delay;
reg [15:0] r_t_bajar_vision;
reg [15:0] r_t_bajar_final;

assign fsm_done_pulse = done_r;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state            <= IDLE;
        pulse            <= PULSE_STOP;
        done_r           <= 0;
        led_debug        <= 0;
        counter          <= 0;
        target           <= 0;
        r_t_subir        <= 0;
        r_t_subir_poco   <= 0;
        r_t_delay        <= 0;
        r_t_bajar_vision <= 0;
        r_t_bajar_final  <= 0;
    end else begin
        done_r <= 0;

        case (state)

            // ───────────────────────────────────
            // IDLE — espera CMD_RACK o CMD_RODILLO
            // ───────────────────────────────────
            IDLE: begin
                pulse   <= PULSE_STOP;
                counter <= 0;

                if (valid) begin
                    r_t_subir        <= t_subir_reg;
                    r_t_subir_poco   <= t_subir_poco_reg;
                    r_t_delay        <= t_delay_reg;
                    r_t_bajar_vision <= t_bajar_vision_reg;
                    r_t_bajar_final  <= t_bajar_final_reg;
                    led_debug        <= ~led_debug;

                    if (cmd_reg == CMD_RODILLO) begin
                        state   <= SUBIR;
                        target  <= t_subir_reg * CICLOS_POR_MS;
                        counter <= 0;
                    end
                    else if (cmd_reg == CMD_RACK) begin
                        state   <= ESPERAR_ALINEACION;
                        counter <= 0;
                    end
                end
            end

            // ───────────────────────────────────
            // SUBIR — solo rack, timer t_subir
            // ───────────────────────────────────
            SUBIR: begin
                pulse <= PULSE_SUBE;
                if (counter >= target) begin
                    pulse   <= PULSE_STOP;
                    state   <= ESPERAR_ALINEACION;
                    counter <= 0;
                end else begin
                    counter <= counter + 1;
                end
            end

            // ───────────────────────────────────
            // ESPERAR_ALINEACION — sin timer
            // espera CMD_ALINEADO (0x21)
            // ───────────────────────────────────
            ESPERAR_ALINEACION: begin
                pulse <= PULSE_STOP;
                if (byte_ready && rx_byte == CMD_ALINEADO) begin
                    state   <= SUBIR_POCO;
                    target  <= r_t_subir_poco * CICLOS_POR_MS;
                    counter <= 0;
                end
            end

            // ───────────────────────────────────
            // SUBIR_POCO — timer t_subir_poco
            // ───────────────────────────────────
            SUBIR_POCO: begin
                pulse <= PULSE_SUBE;
                if (counter >= target) begin
                    pulse   <= PULSE_STOP;
                    state   <= DELAY;
                    target  <= r_t_delay * CICLOS_POR_MS;
                    counter <= 0;
                end else begin
                    counter <= counter + 1;
                end
            end

            // ───────────────────────────────────
            // DELAY — motor detenido, timer t_delay
            // espera a que robot empiece a moverse
            // ───────────────────────────────────
            DELAY: begin
                pulse <= PULSE_STOP;
                if (counter >= target) begin
                    state   <= BAJAR_VISION;
                    target  <= r_t_bajar_vision * CICLOS_POR_MS;
                    counter <= 0;
                end else begin
                    counter <= counter + 1;
                end
            end

            // ───────────────────────────────────
            // BAJAR_VISION — timer t_bajar_vision
            // baja para que lidar tenga visión
            // ───────────────────────────────────
            BAJAR_VISION: begin
                pulse <= PULSE_BAJA;
                if (counter >= target) begin
                    pulse   <= PULSE_STOP;
                    state   <= ESPERAR_WAYPOINT;
                    counter <= 0;
                end else begin
                    counter <= counter + 1;
                end
            end

            // ───────────────────────────────────
            // ESPERAR_WAYPOINT — sin timer
            // espera CMD_WAYPOINT (0x22)
            // ───────────────────────────────────
            ESPERAR_WAYPOINT: begin
                pulse <= PULSE_STOP;
                if (byte_ready && rx_byte == CMD_WAYPOINT) begin
                    state   <= BAJAR_FINAL;
                    target  <= r_t_bajar_final * CICLOS_POR_MS;
                    counter <= 0;
                end
            end

            // ───────────────────────────────────
            // BAJAR_FINAL — timer t_bajar_final
            // deposita el pallet
            // ───────────────────────────────────
            BAJAR_FINAL: begin
                pulse <= PULSE_BAJA;
                if (counter >= target) begin
                    pulse   <= PULSE_STOP;
                    state   <= FSM_DONE;
                    counter <= 0;
                end else begin
                    counter <= counter + 1;
                end
            end

            // ───────────────────────────────────
            // FSM_DONE — manda ACK, regresa a IDLE
            // ───────────────────────────────────
            FSM_DONE: begin
                done_r <= 1;
                state  <= IDLE;
            end

            default: state <= IDLE;

        endcase
    end
end

endmodule